"""ЛР2, пункти 2–4 умови. Аналіз порога зі збережених прогнозів, без навчання й без CSV.

    uv run python analysis.py --saved results --out outputs

Порядок роботи (номери відповідають пунктам умови):
[2] validation: PR- і ROC-криві, ROC-AUC, повна таблиця кандидатів, поріг t*;
[3] перевірка пошуку на наборах A, B, C (checks.py);
[4] лише якщо всі три перевірки пройдено - оцінка зафіксованого t* на test
    і таблиця розібраних помилок.

Мітки й оцінки test використовуються тільки на кроці [4], після фіксації порога.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # збереження графіків у файли без графічного інтерфейсу

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from checks import run_checks
from threshold import COST_FN, COST_FP, Counts, confusion_counts, find_threshold

STANDARD_THRESHOLD = 0.5
THEORETICAL_THRESHOLD = COST_FP / (COST_FP + COST_FN)  # 1/11, лише як орієнтир
ERRORS_PER_TYPE = 5

# Кольори й оформлення графіків.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3de"
CURVE = "#2a78d6"
POINT_STANDARD = "#eb6834"
POINT_CHOSEN = "#1baf7a"


# ---------------------------------------------------------------- допоміжне


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    """CSV без округлення й з однаковим символом кінця рядка на всіх платформах."""
    frame.to_csv(path, index=False, lineterminator="\n")


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def save_json(path: Path, data: dict[str, object]) -> None:
    save_text(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def load_predictions(directory: Path) -> dict[str, np.ndarray]:
    """Ті самі масиви, що зберігає lab2_starter.py (predictions.npz)."""
    with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def split_summary(data: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name in ("train", "validation", "test"):
        y = data[f"{name}_y"]
        positives = int(np.count_nonzero(y == 1))
        rows.append(
            {
                "split": name,
                "n": len(y),
                "positives": positives,
                "positive_fraction": positives / len(y),
            }
        )
    return pd.DataFrame(rows)


def show_threshold(t: float) -> str:
    """Поріг для відображення; у файлах зберігаються неокруглені значення."""
    return "+∞" if math.isinf(t) else f"{t:.6g}"


def show_metric(value: float | None) -> str:
    return "не визначена" if value is None else f"{value:.4f}"


def json_number(t: float) -> float | str:
    return "inf" if math.isinf(t) else float(t)


def comparison_rows(
    y: np.ndarray, s: np.ndarray, rules: list[tuple[str, float]]
) -> pd.DataFrame:
    """Лічильники для кількох порогів прямим застосуванням правила s >= t."""
    rows = []
    for rule, t in rules:
        counts = confusion_counts(y, s, t)
        rows.append(
            {
                "rule": rule,
                "threshold": t,
                "tp": counts.tp,
                "fp": counts.fp,
                "fn": counts.fn,
                "tn": counts.tn,
                "cost": counts.cost,
                "precision": counts.precision,
                "recall": counts.recall,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, headers: dict[str, str]) -> str:
    """Таблиця Markdown зі значеннями, відформатованими для відображення."""
    columns = list(headers)
    lines = [
        "| " + " | ".join(headers[column] for column in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if column in THRESHOLD_COLUMNS:
                cells.append(show_threshold(float(value)))
            elif column in COUNT_COLUMNS:
                cells.append(f"{int(value):,}".replace(",", " "))
            elif column in ("precision", "recall"):
                cells.append(show_metric(None if pd.isna(value) else float(value)))
            elif column == "positive_fraction":
                cells.append(f"{100 * value:.4f} %")
            elif column == "score":
                cells.append(f"{value:.6f}")
            elif column == "score_minus_threshold":
                cells.append(f"{value:+.6f}".replace("-", "−"))
            elif column == "amount":
                cells.append(f"{value:.2f}")
            elif isinstance(value, (bool, np.bool_)):
                cells.append("так" if value else "ні")
            elif isinstance(value, (float, np.floating)) and float(value).is_integer():
                cells.append(str(int(value)))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


THRESHOLD_COLUMNS = ("threshold", "threshold_brute_force", "threshold_cumulative")
COUNT_COLUMNS = ("n", "positives", "tp", "fp", "fn", "tn", "cost", "cost_fast")

COUNT_HEADERS = {
    "rule": "Правило",
    "threshold": "Поріг",
    "tp": "TP",
    "fp": "FP",
    "fn": "FN",
    "tn": "TN",
    "cost": "C",
}


# ------------------------------------------------------------------ графіки


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def mark_point(
    ax: plt.Axes, x: float, y: float, color: str, marker: str, label: str, offset: tuple[int, int]
) -> None:
    ax.plot(
        [x], [y], marker=marker, markersize=10, color=color,
        markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none", zorder=5,
    )
    ax.annotate(
        label, (x, y), xytext=offset, textcoords="offset points", fontsize=9, color=INK,
        ha="left", va="center",
        arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.8,
                    "shrinkA": 4, "shrinkB": 7},
    )


def plot_roc(
    path: Path, y: np.ndarray, s: np.ndarray, auc: float,
    at_standard: Counts, at_chosen: Counts, t_star: float,
) -> None:
    fpr, tpr, _ = roc_curve(y, s)
    negatives = at_standard.fp + at_standard.tn
    positives = at_standard.tp + at_standard.fn
    points = [
        (at_standard, POINT_STANDARD, "o",
         f"t = 0.5\nFP = {at_standard.fp}, FN = {at_standard.fn}"),
        (at_chosen, POINT_CHOSEN, "D",
         f"t* = {show_threshold(t_star)}\nFP = {at_chosen.fp}, FN = {at_chosen.fn}"),
    ]
    zoom = max(4 * at_chosen.fp / negatives, 4 * at_standard.fp / negatives, 0.002)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE)
    for ax, x_max, title in (
        (axes[0], 1.0, "Уся крива"),
        (axes[1], zoom, f"Область малих FPR (до {zoom:.4f})"),
    ):
        style_axes(ax)
        ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1, linestyle="--")
        ax.plot(fpr, tpr, color=CURVE, linewidth=2)
        for counts, color, marker, label in points:
            x, y_value = counts.fp / negatives, counts.tp / positives
            offset = (55, -5) if x_max == 1.0 else (35, -30)
            if marker == "o":
                offset = (55, -45) if x_max == 1.0 else (45, -55)
            mark_point(ax, x, y_value, color, marker, label, offset)
        ax.set_xlim(-0.02 * x_max, x_max)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("FPR = FP / (FP + TN)")
        ax.set_ylabel("TPR = TP / (TP + FN)")
        ax.set_title(title, fontsize=10, color=INK_MUTED, loc="left")
    fig.suptitle(
        f"ROC-крива на validation, ROC-AUC = {auc:.4f}",
        fontsize=12, color=INK, x=0.07, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_pr(
    path: Path, y: np.ndarray, s: np.ndarray, average_precision: float,
    at_standard: Counts, at_chosen: Counts, t_star: float,
) -> None:
    precision_values, recall_values, _ = precision_recall_curve(y, s)
    prevalence = float(np.mean(y == 1))

    fig, ax = plt.subplots(figsize=(6.4, 4.8), facecolor=SURFACE)
    style_axes(ax)
    ax.axhline(prevalence, color=INK_MUTED, linewidth=1, linestyle="--")
    ax.annotate(
        f"частка шахрайських транзакцій {100 * prevalence:.3f} %",
        (0.02, prevalence), xytext=(0, 5), textcoords="offset points",
        fontsize=9, color=INK_MUTED,
    )
    ax.step(recall_values, precision_values, where="post", color=CURVE, linewidth=2)
    for counts, color, marker, label, offset in (
        (at_standard, POINT_STANDARD, "o",
         f"t = 0.5\nprecision = {show_metric(at_standard.precision)}", (-150, -40)),
        (at_chosen, POINT_CHOSEN, "D",
         f"t* = {show_threshold(t_star)}\nprecision = {show_metric(at_chosen.precision)}",
         (-120, -85)),
    ):
        if counts.precision is not None and counts.recall is not None:
            mark_point(ax, counts.recall, counts.precision, color, marker, label, offset)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Повнота (recall) = TP / (TP + FN)")
    ax.set_ylabel("Точність (precision) = TP / (TP + FP)")
    ax.set_title(
        f"PR-крива на validation, AP = {average_precision:.4f}",
        fontsize=12, color=INK, loc="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------- кроки


def validation_step(data: dict[str, np.ndarray], out: Path) -> dict[str, object]:
    """Пункт 2: криві, таблиця кандидатів і вибір t* на validation."""
    y = data["validation_y"]
    s = data["validation_scores"].astype(np.float64)

    table, best = find_threshold(y, s)
    t_star = float(best["threshold"])
    save_csv(table, out / "candidates_validation.csv")

    rules = [
        ("Стандартний поріг", STANDARD_THRESHOLD),
        ("Мінімум вартості", t_star),
        ("Усі негативні", math.inf),
    ]
    comparison = comparison_rows(y, s, rules)
    save_csv(comparison, out / "validation_comparison.csv")

    # Узгодженість прямого застосування правила з таблицею кандидатів.
    at_standard = confusion_counts(y, s, STANDARD_THRESHOLD)
    at_chosen = confusion_counts(y, s, t_star)
    at_infinity = confusion_counts(y, s, math.inf)
    count_columns = ["tp", "fp", "fn", "tn", "cost"]

    def as_list(counts: Counts) -> list[int]:
        return [counts.tp, counts.fp, counts.fn, counts.tn, counts.cost]

    if as_list(at_chosen) != [int(best[column]) for column in count_columns]:
        raise AssertionError("Рядок t* у таблиці не збігається з прямим застосуванням правила.")
    if as_list(at_infinity) != table.iloc[0][count_columns].astype(int).tolist():
        raise AssertionError("Рядок +inf у таблиці не збігається з прямим застосуванням правила.")
    # Правило s >= 0.5 дає ті самі рішення, що й найменший кандидат, не менший за 0.5.
    not_below = table[table["threshold"] >= STANDARD_THRESHOLD]
    if as_list(at_standard) != not_below.iloc[-1][count_columns].astype(int).tolist():
        raise AssertionError("Рядок для 0.5 не збігається з таблицею кандидатів.")
    if not (table[["tp", "fp", "fn", "tn"]].sum(axis=1) == len(y)).all():
        raise AssertionError("Сума лічильників не дорівнює розміру вибірки.")

    auc = float(roc_auc_score(y, s))
    average_precision = float(average_precision_score(y, s))
    plot_roc(out / "roc_validation.png", y, s, auc, at_standard, at_chosen, t_star)
    plot_pr(out / "pr_validation.png", y, s, average_precision, at_standard, at_chosen, t_star)

    minimal = table[table["cost"] == table["cost"].min()]
    reference = comparison_rows(y, s, [("Теоретичний орієнтир 1/11", THEORETICAL_THRESHOLD)])
    smallest_minimal = float(minimal["threshold"].min())
    info = {
        "rule": "s >= t",
        "cost_formula": f"C = {COST_FN} * FN + {COST_FP} * FP",
        "t_star": json_number(t_star),
        "t_star_hex": t_star.hex() if math.isfinite(t_star) else "inf",
        "selected_on": "validation",
        "tie_break": "найбільший поріг серед мінімумів вартості",
        "validation": {
            "n": int(len(y)),
            "positives": int(np.count_nonzero(y == 1)),
            "candidates": int(len(table)),
            "unique_scores": int(len(table) - 1),
            "min_cost": int(best["cost"]),
            "thresholds_with_min_cost": int(len(minimal)),
            "smallest_threshold_with_min_cost": json_number(smallest_minimal),
            "counts_at_t_star": {"tp": at_chosen.tp, "fp": at_chosen.fp,
                                 "fn": at_chosen.fn, "tn": at_chosen.tn},
            "roc_auc": auc,
            "average_precision": average_precision,
        },
        "theoretical_threshold": THEORETICAL_THRESHOLD,
    }
    save_json(out / "threshold.json", info)
    return {
        "table": table,
        "comparison": comparison,
        "reference": reference,
        "t_star": t_star,
        "info": info,
    }


def checks_step(out: Path) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Пункт 3: три малі набори, порівняння прямого перебору з кумулятивними сумами."""
    results = run_checks()
    rows = []
    for result in results:
        save_csv(result["table"], out / f"check_{result['name']}.csv")
        rows.append(
            {
                "set": result["name"],
                "threshold_brute_force": result["slow_threshold"],
                "cost_brute_force": result["slow_cost"],
                "threshold_cumulative": result["fast_threshold"],
                "cost_cumulative": result["fast_cost"],
                "same_counts": result["same_counts"],
                "same_choice": result["same_choice"],
                "matches_manual": result["matches_manual"],
                "passed": result["passed"],
            }
        )
    summary = pd.DataFrame(rows)
    save_csv(summary, out / "checks_summary.csv")
    return results, summary


def test_step(data: dict[str, np.ndarray], t_star: float, out: Path) -> dict[str, object]:
    """Пункт 4: зафіксований t* на test і таблиця розібраних помилок."""
    y = data["test_y"]
    s = data["test_scores"].astype(np.float64)
    rows = data["test_row_ids"]
    amount = data["test_amount"]

    main = comparison_rows(y, s, [("Зафіксований t*", t_star)])
    # Рядки для порівняння на test. Поріг за ними не змінюється.
    reference = comparison_rows(
        y, s, [("Стандартний поріг", STANDARD_THRESHOLD), ("Усі негативні", math.inf)]
    )
    save_csv(pd.concat([main, reference]), out / "test_comparison.csv")

    predicted = s >= t_star
    kinds = {"FP": predicted & (y == 0), "FN": ~predicted & (y == 1)}
    listed = []
    statistics = {}
    for kind, mask in kinds.items():
        frame = pd.DataFrame(
            {
                "csv_row": rows[mask],
                "error": kind,
                "label": y[mask],
                "score": s[mask],
                "amount": amount[mask],
                "score_minus_threshold": s[mask] - t_star,
            }
        ).sort_values("csv_row")
        listed.append(frame.head(ERRORS_PER_TYPE))
        statistics[kind] = {
            "count": int(len(frame)),
            "amount_median": None if frame.empty else float(frame["amount"].median()),
            "amount_min": None if frame.empty else float(frame["amount"].min()),
            "amount_max": None if frame.empty else float(frame["amount"].max()),
            "abs_distance_median": None
            if frame.empty
            else float(frame["score_minus_threshold"].abs().median()),
        }
    errors = pd.concat(listed, ignore_index=True)
    save_csv(errors, out / "test_errors.csv")

    counts = confusion_counts(y, s, t_star)
    metrics = {
        "t_star": json_number(t_star),
        "n": int(len(y)),
        "positives": int(np.count_nonzero(y == 1)),
        "tp": counts.tp, "fp": counts.fp, "fn": counts.fn, "tn": counts.tn,
        "cost": counts.cost,
        "precision": counts.precision,
        "recall": counts.recall,
        "all_errors": statistics,
    }
    save_json(out / "test_metrics.json", metrics)
    return {"main": main, "reference": reference, "errors": errors, "metrics": metrics}


# -------------------------------------------------------------------- звіт


def build_report(
    summary: pd.DataFrame, validation: dict[str, object],
    check_results: list[dict[str, object]], check_summary: pd.DataFrame,
    test: dict[str, object],
) -> str:
    info = validation["info"]["validation"]
    metric_headers = {**COUNT_HEADERS, "precision": "precision", "recall": "recall"}
    parts = [
        "# ЛР2: таблиці, отримані зі збережених прогнозів",
        "",
        "Файл створено скриптом `analysis.py`. Значення округлено для відображення;",
        "неокруглені збережено в CSV і JSON у цій самій теці.",
        "",
        "## Поділ",
        "",
        markdown_table(summary, {"split": "Частина", "n": "Об'єктів",
                                 "positives": "Шахрайських транзакцій",
                                 "positive_fraction": "Частка"}),
        "",
        "## Validation",
        "",
        f"ROC-AUC = {info['roc_auc']:.6f}, AP = {info['average_precision']:.6f}.",
        f"Кандидатів: {info['candidates']:,} ({info['unique_scores']:,} унікальних оцінок"
        " і +∞).".replace(",", " "),
        f"t* = {validation['t_star']!r}, мінімальна вартість C = {info['min_cost']}.",
        f"Порогів із мінімальною вартістю: {info['thresholds_with_min_cost']}.",
        "",
        markdown_table(validation["comparison"], metric_headers),
        "",
        "Додатково (у виборі порога не використовується):",
        "",
        markdown_table(validation["reference"], metric_headers),
        "",
        "## Перевірка пошуку на наборах A, B, C",
        "",
    ]
    for result in check_results:
        parts += [
            f"### Набір {result['name']}",
            "",
            markdown_table(
                result["table"],
                {"threshold": "Поріг", "tp": "TP", "fp": "FP", "fn": "FN", "tn": "TN",
                 "cost": "C (прямий перебір)", "cost_fast": "C (кумулятивні суми)",
                 "match": "Лічильники збігаються"},
            ),
            "",
            f"Прямий перебір: t* = {show_threshold(result['slow_threshold'])},"
            f" C = {result['slow_cost']}."
            f" Кумулятивні суми: t* = {show_threshold(result['fast_threshold'])},"
            f" C = {result['fast_cost']}.",
            "",
        ]
    parts += [
        "### Підсумок перевірок",
        "",
        markdown_table(
            check_summary,
            {"set": "Набір", "threshold_brute_force": "t* (прямий перебір)",
             "cost_brute_force": "C (прямий перебір)",
             "threshold_cumulative": "t* (кумулятивні суми)",
             "cost_cumulative": "C (кумулятивні суми)", "same_counts": "Лічильники збігаються",
             "same_choice": "Поріг збігається", "matches_manual": "Збіг із ручним розрахунком",
             "passed": "Пройдено"},
        ),
        "",
        "## Test",
        "",
        markdown_table(test["main"], metric_headers),
        "",
        "Для порівняння (поріг за цими рядками не змінюється):",
        "",
        markdown_table(test["reference"], metric_headers),
        "",
        "### Перші помилки за номером рядка CSV",
        "",
        markdown_table(
            test["errors"],
            {"csv_row": "Рядок CSV (від 0, без заголовка)", "error": "Тип помилки",
             "label": "Мітка", "score": "Оцінка s", "amount": "Amount",
             "score_minus_threshold": "s − t*"},
        ),
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--saved", type=Path, required=True,
                        help="Тека зі збереженими прогнозами (predictions.npz)")
    parser.add_argument("--out", type=Path, required=True, help="Тека для результатів аналізу")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    data = load_predictions(args.saved)
    summary = split_summary(data)
    save_csv(summary, out / "split_summary.csv")
    print(summary.to_string(index=False))

    print("\n[2] Вибір порога на validation")
    validation = validation_step(data, out)
    info = validation["info"]["validation"]
    print(f"ROC-AUC = {info['roc_auc']:.6f}, AP = {info['average_precision']:.6f}")
    print(f"кандидатів: {info['candidates']}, t* = {validation['t_star']!r},"
          f" C = {info['min_cost']}")
    print(validation["comparison"].to_string(index=False))

    print("\n[3] Перевірка пошуку на наборах A, B, C")
    check_results, check_summary = checks_step(out)
    print(check_summary.to_string(index=False))
    if not bool(check_summary["passed"].all()):
        # Видалення тестових результатів попереднього запуску.
        for name in ("test_comparison.csv", "test_errors.csv", "test_metrics.json", "report.md"):
            (out / name).unlink(missing_ok=True)
        raise SystemExit("Перевірку пошуку порога не пройдено: оцінку на test не виконано.")

    print("\n[4] Оцінка зафіксованого t* на test")
    test = test_step(data, validation["t_star"], out)
    print(test["main"].to_string(index=False))
    print(test["errors"].to_string(index=False))

    report = build_report(summary, validation, check_results, check_summary, test)
    save_text(out / "report.md", report)
    print(f"\nРезультати збережено: {out.resolve()}")


if __name__ == "__main__":
    main()
