"""ЛР2. Порівняння повторного навчання зі збереженим запуском.

    uv run python compare_runs.py results results_rerun

Поділ, мітки й Amount перевіряються на точний збіг. Статистики стандартизації
також перевіряються на точний збіг; якщо на іншій платформі вони відрізняються
в молодших розрядах float64, виводиться найбільша різниця, допуск становить
1e-12. Оцінки після повторного навчання можуть відрізнятися в молодших розрядах
(обчислення PyTorch залежать від версії бібліотеки й платформи), тому для них
виводиться найбільша абсолютна різниця.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load(directory: Path, name: str) -> dict[str, np.ndarray]:
    with np.load(directory / name, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("saved", type=Path, help="Тека збереженого запуску")
    parser.add_argument("rerun", type=Path, help="Тека повторного навчання")
    args = parser.parse_args()

    exact = True
    first = load(args.saved, "predictions.npz")
    second = load(args.rerun, "predictions.npz")
    for key in sorted(first):
        if key.endswith("_scores"):
            continue
        same = key in second and np.array_equal(first[key], second[key])
        exact &= same
        print(f"{key:<22} {'збігається' if same else 'ВІДРІЗНЯЄТЬСЯ'}")

    first_scaler = load(args.saved, "preprocessing.npz")
    second_scaler = load(args.rerun, "preprocessing.npz")
    for key in ("mean", "scale", "var", "n_samples_seen", "features"):
        a, b = first_scaler[key], second_scaler[key]
        if np.array_equal(a, b):
            print(f"scaler.{key:<15} збігається")
        elif a.shape == b.shape and a.dtype.kind == "f" and np.allclose(
            a, b, rtol=1e-12, atol=1e-12
        ):
            difference = np.max(np.abs(a - b))
            print(f"scaler.{key:<15} збігається в межах 1e-12 (|різниця| до {difference:.1e})")
        else:
            exact = False
            print(f"scaler.{key:<15} ВІДРІЗНЯЄТЬСЯ")

    for name in ("validation", "test"):
        a = first[f"{name}_scores"].astype(np.float64)
        b = second[f"{name}_scores"].astype(np.float64)
        print(f"{name}_scores: найбільша |різниця| = {np.max(np.abs(a - b)):.3e}")

    if not exact:
        raise SystemExit("Поділ або статистики стандартизації не відтворились.")
    print("Поділ і статистики стандартизації відтворились.")


if __name__ == "__main__":
    main()
