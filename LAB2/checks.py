"""ЛР2, пункт 3. Незалежна перевірка пошуку порога на малих наборах A, B, C.

Прямий перебір реалізовано окремо від threshold.py: списки Python і пряме
застосування правила s >= t для кожного кандидата, O(n * m), де m - кількість
кандидатів. Він не використовує ні списку кандидатів, ні лічильників, ні
вибраного порога алгоритму з кумулятивними сумами. Алгоритм із кумулятивними
сумами викликається після прямого перебору і тільки для порівняння результатів.

Окремий запуск:
    uv run python checks.py
"""

from __future__ import annotations

import math

import pandas as pd

from threshold import find_threshold

# Набори з умови. У наборі C одинадцять оцінок 0.5 підряд:
# перша з міткою 1, решта десять - з міткою 0.
SETS: dict[str, tuple[list[float], list[int]]] = {
    "A": ([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], [0, 1, 0, 1, 0, 0]),
    "B": ([0.9, 0.6, 0.3], [0, 0, 0]),
    "C": ([0.8] + [0.5] * 11 + [0.2], [1, 1] + [0] * 10 + [0]),
}

# Очікувані пороги й вартості, обчислені вручну (README, розділ 3).
EXPECTED: dict[str, tuple[float, int]] = {
    "A": (0.4, 2),
    "B": (math.inf, 0),
    "C": (0.8, 10),
}

COUNT_COLUMNS = ["tp", "fp", "fn", "tn", "cost"]


def brute_force_rows(scores: list[float], labels: list[int]) -> list[dict[str, float]]:
    """Окремий прохід по всій вибірці за правилом s >= t для кожного кандидата."""
    candidates = [math.inf] + sorted(set(scores), reverse=True)
    rows = []
    for t in candidates:
        tp = fp = fn = tn = 0
        for score, label in zip(scores, labels, strict=True):
            says_fraud = score >= t
            if says_fraud and label == 1:
                tp += 1
            elif says_fraud and label == 0:
                fp += 1
            elif label == 1:
                fn += 1
            else:
                tn += 1
        rows.append(
            {"threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "cost": 10 * fn + fp}
        )
    return rows


def brute_force_choice(rows: list[dict[str, float]]) -> dict[str, float]:
    """Мінімальна вартість; за рівної вартості - більший поріг."""
    best = rows[0]
    for row in rows[1:]:
        cheaper = row["cost"] < best["cost"]
        same_cost_higher = row["cost"] == best["cost"] and row["threshold"] > best["threshold"]
        if cheaper or same_cost_higher:
            best = row
    return best


def check_set(name: str) -> dict[str, object]:
    """Прямий перебір для одного набору та порівняння з алгоритмом із кумулятивними сумами."""
    scores, labels = SETS[name]
    slow = pd.DataFrame(brute_force_rows(scores, labels))
    slow_best = brute_force_choice(slow.to_dict("records"))

    fast, fast_best = find_threshold(labels, scores)

    same_candidates = slow["threshold"].tolist() == fast["threshold"].tolist()
    same_counts = same_candidates and all(
        slow[column].tolist() == fast[column].tolist() for column in COUNT_COLUMNS
    )
    same_choice = (
        slow_best["threshold"] == fast_best["threshold"]
        and slow_best["cost"] == fast_best["cost"]
    )
    expected_threshold, expected_cost = EXPECTED[name]
    matches_manual = (
        slow_best["threshold"] == expected_threshold and slow_best["cost"] == expected_cost
    )

    table = slow.copy()
    if same_candidates:
        for column in COUNT_COLUMNS:
            table[f"{column}_fast"] = fast[column].to_numpy()
        table["match"] = (
            table[COUNT_COLUMNS].to_numpy() == fast[COUNT_COLUMNS].to_numpy()
        ).all(axis=1)
    return {
        "name": name,
        "table": table,
        "slow_threshold": slow_best["threshold"],
        "slow_cost": int(slow_best["cost"]),
        "fast_threshold": float(fast_best["threshold"]),
        "fast_cost": int(fast_best["cost"]),
        "same_candidates": same_candidates,
        "same_counts": same_counts,
        "same_choice": same_choice,
        "matches_manual": matches_manual,
        "passed": same_candidates and same_counts and same_choice and matches_manual,
    }


def run_checks() -> list[dict[str, object]]:
    return [check_set(name) for name in SETS]


def main() -> None:
    results = run_checks()
    for result in results:
        print(f"\nНабір {result['name']}")
        print(result["table"].to_string(index=False))
        print(
            f"  прямий перебір: t* = {result['slow_threshold']}, C = {result['slow_cost']};"
            f" кумулятивні суми: t* = {result['fast_threshold']}, C = {result['fast_cost']}"
        )
        print(f"  перевірку {'пройдено' if result['passed'] else 'НЕ ПРОЙДЕНО'}")
    if not all(result["passed"] for result in results):
        raise SystemExit("Перевірку пошуку порога не пройдено.")
    print("\nУсі три перевірки пройдено.")


if __name__ == "__main__":
    main()
