"""ЛР2. Лічильники матриці плутанини, вартість помилок і пошук порога.

Правило рішення: транзакція класифікується як шахрайська, якщо s >= t.
Вартість помилок: C(t) = 10 * FN(t) + FP(t).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

COST_FN = 10  # ціна пропущеного шахрайства
COST_FP = 1  # ціна хибної тривоги
TABLE_COLUMNS = ["threshold", "tp", "fp", "fn", "tn", "cost"]


@dataclass(frozen=True)
class Counts:
    """Матриця плутанини для одного порога."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def cost(self) -> int:
        return cost(self.fn, self.fp)

    @property
    def precision(self) -> float | None:
        return precision(self.tp, self.fp)

    @property
    def recall(self) -> float | None:
        return recall(self.tp, self.fn)


def cost(fn: int, fp: int) -> int:
    """C = 10 * FN + FP. Amount у вартість не входить."""
    return COST_FN * int(fn) + COST_FP * int(fp)


def precision(tp: int, fp: int) -> float | None:
    """TP / (TP + FP). None, якщо позитивних рішень немає (метрика не визначена)."""
    return None if tp + fp == 0 else tp / (tp + fp)


def recall(tp: int, fn: int) -> float | None:
    """TP / (TP + FN). None, якщо позитивних міток немає (метрика не визначена)."""
    return None if tp + fn == 0 else tp / (tp + fn)


def _as_arrays(y: object, s: object) -> tuple[np.ndarray, np.ndarray]:
    """Мітки як int64, оцінки як float64 (float32 -> float64 без втрати точності)."""
    labels = np.asarray(y)
    scores = np.asarray(s, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("Мітки й оцінки мають бути одновимірними масивами однакової довжини.")
    if len(scores) == 0:
        raise ValueError("Порожня вибірка.")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Мітки мають бути 0 або 1.")
    if not np.isfinite(scores).all():
        raise ValueError("Оцінки мають бути скінченними числами.")
    return labels.astype(np.int64), scores


def confusion_counts(y: object, s: object, t: float) -> Counts:
    """Пряме застосування правила s >= t для одного порога: один прохід, O(n)."""
    labels, scores = _as_arrays(y, s)
    predicted = scores >= t  # для t = +inf усі рішення негативні
    actual = labels == 1
    return Counts(
        tp=int(np.count_nonzero(predicted & actual)),
        fp=int(np.count_nonzero(predicted & ~actual)),
        fn=int(np.count_nonzero(~predicted & actual)),
        tn=int(np.count_nonzero(~predicted & ~actual)),
    )


def candidate_table(y: object, s: object) -> pd.DataFrame:
    """Таблиця всіх кандидатів: +inf і кожна унікальна оцінка, за спаданням порога.

    Одне сортування O(n log n), далі кумулятивні суми O(n); повторного
    проходу по вибірці для кожного порога немає.

    Після сортування за спаданням оцінок для порога t, що дорівнює унікальній
    оцінці, позитивне рішення отримують приклади до останнього прикладу групи
    з цією оцінкою включно. Тому TP і FP для цього порога дорівнюють
    кумулятивним сумам у кінці групи, і приклади з рівними оцінками отримують
    однакове рішення.
    """
    labels, scores = _as_arrays(y, s)
    order = np.argsort(scores, kind="stable")[::-1]  # за спаданням оцінок
    s_sorted = scores[order]
    y_sorted = labels[order]

    cum_tp = np.cumsum(y_sorted == 1)  # позитивних серед перших k прикладів
    cum_fp = np.cumsum(y_sorted == 0)  # негативних серед перших k прикладів
    positives = int(cum_tp[-1])
    negatives = int(cum_fp[-1])

    # Індекс останнього елемента кожної групи рівних оцінок.
    is_group_end = np.append(s_sorted[1:] != s_sorted[:-1], True)
    group_end = np.flatnonzero(is_group_end)

    # Перший рядок - поріг +inf: жодного позитивного рішення.
    threshold = np.concatenate(([np.inf], s_sorted[group_end]))
    tp = np.concatenate(([0], cum_tp[group_end])).astype(np.int64)
    fp = np.concatenate(([0], cum_fp[group_end])).astype(np.int64)
    fn = positives - tp
    tn = negatives - fp
    return pd.DataFrame(
        {
            "threshold": threshold,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "cost": COST_FN * fn + COST_FP * fp,
        },
        columns=TABLE_COLUMNS,
    )


def select_threshold(table: pd.DataFrame) -> pd.Series:
    """Рядок із мінімальною вартістю; серед кількох мінімумів - найбільший поріг."""
    minimal = table[table["cost"] == table["cost"].min()]
    return minimal.loc[minimal["threshold"].idxmax()]


def find_threshold(y: object, s: object) -> tuple[pd.DataFrame, pd.Series]:
    """Повна таблиця кандидатів і вибраний рядок t*."""
    table = candidate_table(y, s)
    return table, select_threshold(table)
