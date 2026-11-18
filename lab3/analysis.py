"""ЛР3, пункти 3-5 умови. Аналіз зі збережених журналів і одноразова тестова оцінка.

    uv run python analysis.py --data /шлях/до/набору [--runs runs] [--model lab3_model.pt] [--out outputs]

Порядок роботи:
[1] журнали семи запусків; перевірка, що SHA-256 split.csv однаковий у всіх і збігається
    з поточним файлом; таблиця запусків і криві;
[2] розклад ефектів: середні, розмахи, R, Δpre, Δunf;
[3] перевірка, що lab3_model.pt відповідає правилу вибору за журналами;
[4] лише після цього - логіти test на CPU, macro-F1, матриця плутанини, п'ять помилок.

Код, що завантажує test, викликається тільки тут. Результати - у outputs/ і outputs/report.md.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import MaxNLocator
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from torchvision.models import resnet18

from data import CLASSES, PART_DIR, SPLIT_PATH, load_split, make_loader
from train import RUNS, choose, evaluate, macro_f1, run_name, sha256_of

N_ERRORS = 5
COLORS = {"scratch": "#52514e", "frozen": "#2a78d6", "unfreeze": "#eb6834"}
LINESTYLES = {1: "-", 2: "--", 3: ":"}
GRID = "#e4e3de"


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def save_csv(frame: pd.DataFrame, path: Path, **kwargs) -> None:
    frame.to_csv(path, lineterminator="\n", **kwargs)


# ---------------------------------------------------------------- [1] журнали


def load_logs(runs: Path) -> list[dict[str, object]]:
    """Сім журналів у порядку таблиці умови. Зупиняється, якщо хеші поділу не збігаються."""
    logs = [json.loads((runs / run_name(regime, seed) / "run.json").read_text(encoding="utf-8")) for regime, seed in RUNS]
    current = sha256_of(SPLIT_PATH)
    in_logs = {log["split_sha256"] for log in logs}
    if in_logs != {current}:
        raise SystemExit(f"SHA-256 split.csv у журналах {sorted(in_logs)} не збігається з поточним {current}.")
    return logs


def runs_table(logs: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame({
        "#": range(1, len(logs) + 1),
        "regime": [log["regime"] for log in logs],
        "seed": [log["seed"] for log in logs],
        "train_loss_epoch8": [log["train_loss"][-1] for log in logs],
        "val_macro_f1_epoch8": [log["val_macro_f1"][-1] for log in logs],
        "train_time_s": [log["train_time_s"] for log in logs],
    })


def runs_markdown(table: pd.DataFrame) -> str:
    lines = ["| # | Режим | Зерно | Втрата train, епоха 8 | macro-F1 validation, епоха 8 | Час навчання, с |",
             "|---|---|---|---|---|---|"]
    for r in table.itertuples(index=False):
        lines.append(f"| {r[0]} | {r.regime} | {r.seed} | {r.train_loss_epoch8:.4f} | {r.val_macro_f1_epoch8:.4f} | {r.train_time_s:.1f} |")
    return "\n".join(lines)


def plot_curves(logs: list[dict[str, object]], path: Path) -> None:
    """Дві панелі на спільних осях: втрата на train і macro-F1 на validation за епохами, сім запусків."""
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(12, 4.8))
    for log in logs:
        epochs = range(1, len(log["train_loss"]) + 1)
        style = {"color": COLORS[log["regime"]], "linestyle": LINESTYLES[log["seed"]], "marker": "o",
                 "markersize": 3.5, "linewidth": 1.8, "label": f"{log['regime']}, зерно {log['seed']}"}
        ax_loss.plot(epochs, log["train_loss"], **style)
        ax_f1.plot(epochs, log["val_macro_f1"], **style)
    for ax, title, ylabel in ((ax_loss, "Втрата на train", "втрата"), (ax_f1, "macro-F1 на validation", "macro-F1")):
        ax.set_title(title)
        ax.set_xlabel("епоха")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, color=GRID)
        ax.set_axisbelow(True)
    ax_loss.set_ylim(bottom=0)
    ax_f1.set_ylim(0, 1)
    ax_f1.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- [2] ефекти


def effects(logs: list[dict[str, object]]) -> dict[str, float]:
    """Середнє і розмах за зернами для frozen і unfreeze, поріг шуму R, Δpre і Δunf."""
    final = {}
    for log in logs:
        final.setdefault(log["regime"], []).append(log["val_macro_f1"][-1])
    f_scratch = final["scratch"][0]
    mean = {r: float(np.mean(v)) for r, v in final.items()}
    spread = {r: float(max(v) - min(v)) for r, v in final.items() if len(v) > 1}
    noise = max(spread.values())
    return {
        "F_scratch_1": f_scratch,
        "F_frozen_mean": mean["frozen"], "R_frozen": spread["frozen"],
        "F_unfreeze_mean": mean["unfreeze"], "R_unfreeze": spread["unfreeze"],
        "R": noise,
        "delta_pre": mean["frozen"] - f_scratch,
        "delta_unf": mean["unfreeze"] - mean["frozen"],
        "delta_total": mean["unfreeze"] - f_scratch,
    }


def verdict(delta: float, noise: float) -> str:
    if abs(delta) > noise:
        return "|Δ| > R: ефект вартий перевірки, але не доведений"
    return "|Δ| ≤ R: різниця не відрізняється від розкиду, результату немає"


def effects_markdown(e: dict[str, float]) -> str:
    rows = [
        ("F_scratch,1", f"{e['F_scratch_1']:.4f}"),
        ("F̄_frozen, R_frozen", f"{e['F_frozen_mean']:.4f}, {e['R_frozen']:.4f}"),
        ("F̄_unfreeze, R_unfreeze", f"{e['F_unfreeze_mean']:.4f}, {e['R_unfreeze']:.4f}"),
        ("R = max(R_frozen, R_unfreeze)", f"{e['R']:.4f}"),
        ("Δpre = F̄_frozen − F_scratch,1", f"{e['delta_pre']:+.4f}"),
        ("Δunf = F̄_unfreeze − F̄_frozen", f"{e['delta_unf']:+.4f}"),
        ("Δpre + Δunf", f"{e['delta_total']:+.4f}"),
    ]
    lines = ["| Величина | Значення |", "|---|---|"] + [f"| {name} | {value} |" for name, value in rows]
    lines += ["", f"Δpre: {verdict(e['delta_pre'], e['R'])}.", f"Δunf: {verdict(e['delta_unf'], e['R'])}."]
    return "\n".join(lines)


# ---------------------------------------------------------------- [3] вибір


def load_selected(path: Path) -> tuple[torch.nn.Module, dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu")
    model = resnet18(num_classes=len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint


def check_selection(logs, checkpoint, model, root: Path, split, runs: Path) -> list[str]:
    """Правило вибору за журналами проти вмісту lab3_model.pt; логіти validation моделі проти збережених."""
    regime, seed = choose(logs)
    saved = (checkpoint["regime"], checkpoint["seed"])
    if saved != (regime, seed):
        raise SystemExit(f"lab3_model.pt містить {saved}, а правило вибору дає {(regime, seed)}. Test не обчислено.")
    logits, _ = evaluate(model, make_loader(root, split, "validation"), torch.device("cpu"))
    stored = np.load(runs / run_name(regime, seed) / "val_logits.npy")
    diff = float(np.abs(logits - stored).max())
    return [f"Правило вибору за журналами: {regime}, зерно {seed}. У `lab3_model.pt`: {saved[0]}, зерно {saved[1]}, "
            f"класи {checkpoint['classes']}, img_size {checkpoint['img_size']}.",
            f"Найбільша різниця між логітами validation цієї моделі на CPU і збереженими в журналі: {diff:.2e}."]


# ---------------------------------------------------------------- [4] test


def confusion_markdown(cm: np.ndarray) -> str:
    lines = ["| справжній \\ передбачений | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    for name, row in zip(CLASSES, cm, strict=True):
        lines.append(f"| {name} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def plot_confusion(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Матриця плутанини на test")
    ax.set_xlabel("передбачений клас")
    ax.set_ylabel("справжній клас")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def frequent_pairs(cm: np.ndarray) -> list[tuple[int, int, int]]:
    """Позадіагональні пари (кількість, справжній, передбачений) за спаданням кількості, далі за індексами."""
    pairs = [(int(cm[t, p]), t, p) for t in range(len(CLASSES)) for p in range(len(CLASSES)) if t != p and cm[t, p] > 0]
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    return pairs


def characteristic_errors(cm: np.ndarray, files: np.ndarray, labels: np.ndarray, logits: np.ndarray) -> pd.DataFrame:
    """П'ять помилок: для кожної з найчастіших пар - перший за іменем файл; якщо пар менше п'яти - другі файли і т. д."""
    pred = logits.argmax(axis=1)
    prob = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    position = {file: i for i, file in enumerate(files)}
    rows: list[dict[str, object]] = []
    rank = 0
    pairs = frequent_pairs(cm)
    while len(rows) < N_ERRORS and rank < int(cm.max()):
        for count, t, p in pairs:
            candidates = sorted(files[(labels == t) & (pred == p)])
            if rank < len(candidates) and len(rows) < N_ERRORS:
                file = candidates[rank]
                rows.append({"file": file, "true": CLASSES[t], "predicted": CLASSES[p],
                             "probability": float(prob[position[file], p]), "count": count})
        rank += 1
    return pd.DataFrame(rows)


def errors_markdown(errors: pd.DataFrame) -> str:
    lines = ["| Файл | Справжній клас | Передбачений клас | Імовірність передбаченого класу | Кількість таких помилок у test |",
             "|---|---|---|---|---|"]
    for r in errors.itertuples(index=False):
        lines.append(f"| {r.file} | {r.true} | {r.predicted} | {r.probability:.4f} | {r.count} |")
    return "\n".join(lines)


def plot_errors(errors: pd.DataFrame, root: Path, path: Path) -> None:
    fig, axes = plt.subplots(1, len(errors), figsize=(3.2 * len(errors), 4.2))
    for ax, r in zip(np.atleast_1d(axes), errors.itertuples(index=False), strict=True):
        ax.imshow(Image.open(root / PART_DIR["test"] / r.true / r.file).convert("RGB"))
        ax.set_title(f"{r.file}\n{r.true} → {r.predicted}\np = {r.probability:.2f}", fontsize=10)
        ax.axis("off")
    fig.suptitle("П'ять характерних помилок на test (справжній → передбачений клас)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- команда


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="тека з seg_train/ і seg_test/")
    parser.add_argument("--runs", type=Path, default=Path("runs"), help="тека журналів запусків")
    parser.add_argument("--model", type=Path, default=Path("lab3_model.pt"), help="файл вибраної моделі")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="тека результатів аналізу")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    split = load_split()
    report: list[str] = ["# Результати аналізу (пункти 3-5)", ""]

    logs = load_logs(args.runs)
    commits = sorted({(log["git_commit"], log["git_uncommitted_changes"]) for log in logs})
    env = logs[0]
    report += ["## Сім запусків", "",
               f"SHA-256 `split.csv` однаковий у семи журналах і збігається з поточним файлом: `{env['split_sha256']}`.",
               f"Коміт коду (незакомічені зміни): {', '.join(f'`{c}` ({d})' for c, d in commits)}.",
               f"Python {env['python']}, PyTorch {env['torch']}, torchvision {env['torchvision']}, "
               f"пристрій `{env['device']}` ({env['device_name']}), потоків {env['torch_num_threads']}.", ""]
    table = runs_table(logs)
    save_csv(table, args.out / "runs_table.csv", index=False)
    plot_curves(logs, args.out / "curves.png")
    report += [runs_markdown(table), "", "![Криві семи запусків](curves.png)", ""]

    e = effects(logs)
    save_json(args.out / "effects.json", e)
    report += ["## Розклад ефектів", "", effects_markdown(e), ""]

    model, checkpoint = load_selected(args.model)
    report += ["## Перевірка вибору", ""] + check_selection(logs, checkpoint, model, args.data, split, args.runs)
    if abs(e["delta_unf"]) <= e["R"]:
        report.append("|Δunf| ≤ R, тому вибір між frozen і unfreeze визначив шум.")
    report.append("")

    logits, labels = evaluate(model, make_loader(args.data, split, "test"), torch.device("cpu"))
    pred = logits.argmax(axis=1)
    selected_f1 = next(log["val_macro_f1"][-1] for log in logs
                       if (log["regime"], log["seed"]) == (checkpoint["regime"], checkpoint["seed"]))
    test_f1 = macro_f1(logits, labels)
    accuracy = float(accuracy_score(labels, pred))
    se = math.sqrt(accuracy * (1 - accuracy) / len(labels))
    metrics = {"regime": checkpoint["regime"], "seed": checkpoint["seed"], "n_test": int(len(labels)),
               "val_macro_f1": selected_f1, "test_macro_f1": test_f1, "test_minus_val": test_f1 - selected_f1,
               "accuracy": accuracy, "se_accuracy": se}
    save_json(args.out / "test_metrics.json", metrics)
    report += ["## Тестова оцінка (один раз, CPU)", "",
               "| Величина | Значення |", "|---|---|",
               f"| macro-F1 на test | {test_f1:.4f} |",
               f"| F_r,s вибраного запуску на validation | {selected_f1:.4f} |",
               f"| test − validation | {test_f1 - selected_f1:+.4f} |",
               f"| accuracy на test, p | {accuracy:.4f} |",
               f"| √(p(1−p)/n), n = {len(labels)} | {se:.4f} |", ""]

    cm = confusion_matrix(labels, pred, labels=range(len(CLASSES)))
    save_csv(pd.DataFrame(cm, index=CLASSES, columns=CLASSES), args.out / "confusion_matrix.csv")
    plot_confusion(cm, args.out / "confusion_matrix.png")
    report += ["### Матриця плутанини", "", confusion_markdown(cm), "", "![Матриця плутанини](confusion_matrix.png)", ""]

    files = split.loc[split["part"] == "test", "file"].to_numpy()
    errors = characteristic_errors(cm, files, labels, logits)
    save_csv(errors, args.out / "test_errors.csv", index=False)
    plot_errors(errors, args.data, args.out / "test_errors.png")
    report += ["### П'ять характерних помилок", "", errors_markdown(errors), "", "![П'ять помилок](test_errors.png)", ""]

    text = "\n".join(report)
    (args.out / "report.md").write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()