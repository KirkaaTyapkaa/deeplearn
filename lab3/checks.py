"""ЛР3, пункти 1-2 умови. Перевірки до навчання та перевірка режимів навчання.

    uv run python checks.py --data /шлях/до/набору [--out outputs]

[1] рисунок перших 16 зображень навчального пакета після перетворень; початкова втрата L0
    для scratch і frozen на першому пакеті до жодного кроку оптимізатора;
[2] таблиця параметрів трьох режимів; що змінюється за 5 кроків у frozen і unfreeze;
    дослід із навмисною помилкою - frozen із model.train() замість set_train_mode.

Усі моделі тут будуються зі зерном 1 і після перевірки відкидаються: прямий прохід у режимі
навчання оновлює ковзні статистики BN, тому ці екземпляри в запусках не використовуються.
Результати зберігаються в outputs/checks.md і outputs/train_batch.png.
"""

from __future__ import annotations

import argparse
import math
from itertools import islice
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import CLASSES, IMAGENET_MEAN, IMAGENET_STD, load_split, make_loader
from model import REGIMES, TOP_LEVEL, Regime, all_regimes_summary, build_model, pick_device, set_train_mode, summary_markdown

CHECK_SEED = 1
CHECK_STEPS = 5
BN_LAYER = "layer1.0.bn1"
RUNNING_STATS = ("running_mean", "running_var")


def yes(flag: bool | None) -> str:
    if flag is None:
        return "немає BN"
    return "так" if flag else "ні"


# ---------------------------------------------------------------- [1] пакет і L0


def save_batch_figure(loader: DataLoader, path: Path) -> None:
    """Перші 16 зображень пакета в тому вигляді, в якому їх отримує модель, з оберненим нормуванням."""
    images, labels = next(iter(loader))
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8.5))
    for ax, image, label in zip(axes.flat, images[:16], labels[:16], strict=True):
        ax.imshow((image * std + mean).clamp(0, 1).permute(1, 2, 0))
        ax.set_title(CLASSES[label], fontsize=10)
        ax.axis("off")
    fig.suptitle("Навчальний пакет після перетворень (перші 16 зображень)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def initial_loss(regime: Regime, root: Path, split, device: torch.device) -> float:
    """Втрата на першому пакеті навчального завантажувача після set_train_mode, без кроку оптимізатора."""
    model, _ = build_model(regime, CHECK_SEED, device)
    images, labels = next(iter(make_loader(root, split, "train")))
    set_train_mode(model, regime)
    with torch.no_grad():
        loss = nn.CrossEntropyLoss()(model(images.to(device)), labels.to(device))
    return loss.item()


# ---------------------------------------------------------------- [2] що змінюється


def snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    """Копія параметрів і буферів (running_mean, running_var, num_batches_tracked)."""
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def train_steps(regime: Regime, root: Path, split, device: torch.device, use_set_train_mode: bool):
    """Новий екземпляр зі зерном 1, 5 кроків оптимізатора на перших 5 пакетах. Повертає стани до і після."""
    model, optimizer = build_model(regime, CHECK_SEED, device)
    before = snapshot(model)
    if use_set_train_mode:
        set_train_mode(model, regime)
    else:
        model.train()  # навмисна помилка: BN заморожених модулів лишається в режимі навчання
    loss_fn = nn.CrossEntropyLoss()
    for images, labels in islice(make_loader(root, split, "train"), CHECK_STEPS):
        optimizer.zero_grad()
        loss_fn(model(images.to(device)), labels.to(device)).backward()
        optimizer.step()
    return before, snapshot(model)


def module_changes(before: dict, after: dict, module: str) -> tuple[bool, bool | None]:
    """Чи змінився хоч один параметр модуля і хоч одна ковзна статистика BN у ньому (точне порівняння)."""
    keys = [k for k in before if k.split(".")[0] == module]
    stat_keys = [k for k in keys if k.endswith(RUNNING_STATS)]
    param_keys = [k for k in keys if not k.endswith(RUNNING_STATS) and not k.endswith("num_batches_tracked")]
    params_changed = any(not torch.equal(before[k], after[k]) for k in param_keys)
    stats_changed = any(not torch.equal(before[k], after[k]) for k in stat_keys) if stat_keys else None
    return params_changed, stats_changed


def changes_table(regime: Regime, before: dict, after: dict) -> list[dict[str, object]]:
    """Рядок на кожен модуль верхнього рівня. Очікування - для правильного режиму (набір set_train_mode)."""
    rows = []
    for module in TOP_LEVEL:
        params_changed, stats_changed = module_changes(before, after, module)
        trainable = module in regime.trainable
        expected_stats = None if stats_changed is None else trainable
        rows.append({
            "regime": regime.name,
            "module": module,
            "params": params_changed,
            "stats": stats_changed,
            "expected": params_changed == trainable and stats_changed == expected_stats,
        })
    return rows


def changes_markdown(rows: list[dict[str, object]]) -> str:
    lines = ["| Режим | Модуль | Параметри змінилися | Ковзні статистики BN змінилися | Збігається з очікуваним |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['regime']} | `{r['module']}` | {yes(r['params'])} | {yes(r['stats'])} | {yes(r['expected'])} |")
    return "\n".join(lines)


def bn_shift(before: dict, after: dict, layer: str = BN_LAYER) -> dict[str, float]:
    """Максимальна абсолютна зміна running_mean і running_var у вказаному шарі BN."""
    return {stat: (after[f"{layer}.{stat}"] - before[f"{layer}.{stat}"]).abs().max().item() for stat in RUNNING_STATS}


# ---------------------------------------------------------------- звіт


def run_checks(root: Path, out: Path, device: torch.device) -> bool:
    """Виконує всі перевірки, пише outputs/checks.md, повертає True, якщо все збіглося з очікуваним."""
    out.mkdir(parents=True, exist_ok=True)
    split = load_split()
    report: list[str] = ["# Перевірки до навчання (пункти 1-2)", "", f"Пристрій: `{device}`, зерно {CHECK_SEED}.", ""]

    torch.manual_seed(CHECK_SEED)
    save_batch_figure(make_loader(root, split, "train"), out / "train_batch.png")
    report += ["## Пакет після перетворень", "", "![Навчальний пакет](train_batch.png)", ""]

    report += ["## Початкова втрата", "", f"ln 6 = {math.log(6):.4f}", "",
               "| Режим | L0 | L0 - ln 6 |", "|---|---|---|"]
    for name in ("scratch", "frozen"):
        loss = initial_loss(REGIMES[name], root, split, device)
        report.append(f"| {name} | {loss:.4f} | {loss - math.log(6):+.4f} |")
    report.append("")

    summaries = all_regimes_summary(CHECK_SEED, device)
    params_ok = all(s["matches"] and s["frozen_in_optimizer"] == 0 and s["optimizer_covers_trainable"] for s in summaries)
    report += ["## Параметри та групи оптимізатора", "", summary_markdown(summaries), ""]

    changes_ok = True
    report += [f"## Що змінюється за {CHECK_STEPS} кроків із set_train_mode", ""]
    all_rows = []
    for name in ("frozen", "unfreeze"):
        before, after = train_steps(REGIMES[name], root, split, device, use_set_train_mode=True)
        rows = changes_table(REGIMES[name], before, after)
        changes_ok &= all(r["expected"] for r in rows)
        all_rows += rows
    report += [changes_markdown(all_rows), ""]

    before, after = train_steps(REGIMES["frozen"], root, split, device, use_set_train_mode=False)
    shift = bn_shift(before, after)
    report += [f"## Дослід із навмисною помилкою: frozen із model.train()", "",
               changes_markdown(changes_table(REGIMES["frozen"], before, after)), "",
               f"Максимальна абсолютна зміна в `{BN_LAYER}` після {CHECK_STEPS} кроків: "
               f"running_mean {shift['running_mean']:.6f}, running_var {shift['running_var']:.6f}.", ""]

    passed = params_ok and changes_ok
    report += [f"Усі перевірки збіглися з очікуваним: {yes(passed)}", ""]
    text = "\n".join(report)
    (out / "checks.md").write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="тека з seg_train/ і seg_test/")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="тека для рисунка і checks.md")
    args = parser.parse_args()
    if not run_checks(args.data, args.out, pick_device()):
        raise SystemExit("Перевірки не збіглися з очікуваним.")


if __name__ == "__main__":
    main()