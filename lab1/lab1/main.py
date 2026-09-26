"""
uv run python main.py        - правильна реалізація
uv run python main.py --bug  - дослід: без ділення градієнта за логітами на N
"""

import argparse
from pathlib import Path

import numpy as np

from checks import compare_with_torch, numeric_check
from data import load_data
from model import backward, forward, init_params, log_softmax
from torch_reference import torch_loss_and_grads


def yes(ok):
    return "так" if ok else "ні"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bug", action="store_true",
                        help="прибрати ділення на N у градієнті за логітами")
    args = parser.parse_args()

    X, y, _, _ = load_data()
    params = init_params()

    loss_np, cache = forward(params, X, y)
    grads_np = backward(params, cache, divide_by_n=not args.bug)
    loss_t, grads_t = torch_loss_and_grads(params, X, y)

    out = [f"# Результати ({'з помилкою' if args.bug else 'правильна реалізація'})", ""]

    out += ["## Звірка з PyTorch", "",
            f"Втрата NumPy: {loss_np!r}  ", f"Втрата PyTorch: {loss_t!r}", "",
            "| Величина | Максимальна абсолютна різниця NumPy / PyTorch | Перевірку пройдено |",
            "|---|---|---|"]
    torch_rows = compare_with_torch(loss_np, grads_np, loss_t, grads_t)
    for name, d, ok in torch_rows:
        out.append(f"| {name} | {d:.3e} | {yes(ok)} |")

    out += ["", "## Чисельна перевірка (eps = 1e-6, допуск 1e-7)", "",
            "| Параметр | Градієнт backward() | Чисельна похідна | Абсолютна різниця | Перевірку пройдено |",
            "|---|---|---|---|---|"]
    num_rows, relu_ok = numeric_check(params, X, y, grads_np)
    for name, g_man, g_num, d, ok in num_rows:
        out.append(f"| {name} | {g_man:.10e} | {g_num:.10e} | {d:.3e} | {yes(ok)} |")
    out += ["", f"Знаки передактивацій ReLU при збуренні не змінюються: {yes(relu_ok)}", ""]

    if args.bug:
        out += ["Відношення градієнтів з помилкою до правильних (PyTorch):", ""]
        for k in ("W1", "b1", "W2", "b2"):
            r = grads_np[k] / grads_t[k]
            out.append(f"- {k}: від {r.min():.12f} до {r.max():.12f}")
        out.append("")
    else:
        # перевірка стабільності log-softmax на великих логітах
        out += ["## Стабільність log-softmax", "",
                "| Логіти | Клас | Стабільна CE | Наївна CE |", "|---|---|---|---|"]
        for z, c in (([1000.0, 0.0, 0.0], 1), ([0.0, -1000.0, 0.0], 1)):
            Z = np.array([z])
            stable = -log_softmax(Z)[0, c]
            with np.errstate(all="ignore"):
                naive = -np.log(np.exp(Z[0, c]) / np.exp(Z).sum())
            out.append(f"| {z} | {c} | {stable:.4f} | {naive} |")
        out.append("")

    text = "\n".join(out)
    print(text)
    Path("results").mkdir(exist_ok=True)
    Path("results", "bug.md" if args.bug else "correct.md").write_text(text, encoding="utf-8")

    passed = all(r[-1] for r in torch_rows) and all(r[-1] for r in num_rows)
    if args.bug:
        print("Помилку виявлено:", yes(not passed))
    else:
        print("Усі перевірки пройдено:", yes(passed))


if __name__ == "__main__":
    main()
