"""ЛР3. Модель: ResNet-18 у трьох режимах навчання, групи AdamW і режими модулів.

Окремий запуск друкує таблицю параметрів для трьох режимів (пункт 2 умови):
    uv run python model.py

Порядок побудови моделі в кожному запуску (кроки 1-5 умови):
    1) torch.manual_seed(seed), np.random.seed(seed);
    2) resnet18 з вагами ImageNet або випадковими;
    3) model.fc = Linear(512, 6) зі стандартною ініціалізацією;
    4) requires_grad=False для всього, крім модулів, що навчаються;
    5) перенесення на пристрій і створення оптимізатора.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

NUM_CLASSES = 6
LR_HEAD = 3e-3
LR_LAYER4 = 3e-4
WEIGHT_DECAY = 1e-4
EXPECTED_TOTAL = 11_179_590

# Модулі верхнього рівня ResNet-18 у порядку прямого проходу; fc - голова, решта - кістяк.
TOP_LEVEL = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc")
BACKBONE = TOP_LEVEL[:-1]


@dataclass(frozen=True)
class Regime:
    """Режим навчання: звідки ваги і які модулі навчаються, з якою швидкістю.

    groups - групи AdamW у вигляді (модулі, швидкість навчання). Модулі кістяка, яких немає
    в жодній групі, заморожені: requires_grad=False і eval() під час навчання.
    """

    name: str
    pretrained: bool
    groups: tuple[tuple[tuple[str, ...], float], ...]
    expected_trainable: int

    @property
    def trainable(self) -> tuple[str, ...]:
        return tuple(module for modules, _ in self.groups for module in modules)

    @property
    def frozen(self) -> tuple[str, ...]:
        return tuple(module for module in BACKBONE if module not in self.trainable)


REGIMES: dict[str, Regime] = {
    "scratch": Regime("scratch", False, ((TOP_LEVEL, LR_HEAD),), 11_179_590),
    "frozen": Regime("frozen", True, ((("fc",), LR_HEAD),), 3_078),
    "unfreeze": Regime("unfreeze", True, ((("fc",), LR_HEAD), (("layer4",), LR_LAYER4)), 8_396_806),
}


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- побудова


def build_model(regime: Regime, seed: int, device: torch.device) -> tuple[nn.Module, torch.optim.AdamW]:
    """Кроки 1-5 умови. Повертає модель на пристрої та оптимізатор лише з параметрів, що навчаються."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    weights = ResNet18_Weights.IMAGENET1K_V1 if regime.pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(512, NUM_CLASSES)

    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.split(".")[0] in regime.trainable)

    model.to(device)
    param_groups = [
        {"params": [p for module in modules for p in getattr(model, module).parameters()], "lr": lr}
        for modules, lr in regime.groups
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    return model, optimizer


def set_train_mode(model: nn.Module, regime: Regime) -> None:
    """model.train(), потім eval() для заморожених модулів: їхня BN лишається зі статистиками ImageNet."""
    model.train()
    for module in regime.frozen:
        getattr(model, module).eval()


# ---------------------------------------------------------------- параметри


def numel(parameters) -> int:
    return sum(p.numel() for p in parameters)


def parameter_summary(regime: Regime, model: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, object]:
    """Загальна кількість, кількість із requires_grad=True, склад груп і перевірки збігу з таблицею режимів."""
    in_optimizer = [p for group in optimizer.param_groups for p in group["params"]]
    total = numel(model.parameters())
    trainable = numel(p for p in model.parameters() if p.requires_grad)
    groups = [(numel(group["params"]), group["lr"]) for group in optimizer.param_groups]
    return {
        "regime": regime.name,
        "total": total,
        "trainable": trainable,
        "groups": groups,
        "frozen_in_optimizer": sum(not p.requires_grad for p in in_optimizer),
        "optimizer_covers_trainable": numel(in_optimizer) == trainable,
        "matches": total == EXPECTED_TOTAL and trainable == regime.expected_trainable,
    }


def summary_markdown(summaries: list[dict[str, object]]) -> str:
    lines = [
        "| Режим | Усього параметрів | requires_grad=True | Групи AdamW (параметрів, швидкість) | Збіг |",
        "|---|---|---|---|---|",
    ]
    for s in summaries:
        groups = "; ".join(f"{n:,} при {lr:g}".replace(",", " ") for n, lr in s["groups"])
        ok = s["matches"] and s["frozen_in_optimizer"] == 0 and s["optimizer_covers_trainable"]
        lines.append(
            f"| {s['regime']} | {s['total']:,} | {s['trainable']:,} | {groups} | {'так' if ok else 'ні'} |"
            .replace(",", " ")
        )
    return "\n".join(lines)


def all_regimes_summary(seed: int = 1, device: torch.device | None = None) -> list[dict[str, object]]:
    device = device or torch.device("cpu")
    summaries = []
    for regime in REGIMES.values():
        model, optimizer = build_model(regime, seed, device)
        summaries.append(parameter_summary(regime, model, optimizer))
    return summaries


def main() -> None:
    summaries = all_regimes_summary()
    print(summary_markdown(summaries))
    passed = all(
        s["matches"] and s["frozen_in_optimizer"] == 0 and s["optimizer_covers_trainable"] for s in summaries
    )
    print(f"\nЗаморожених параметрів в оптимізаторі: {sum(s['frozen_in_optimizer'] for s in summaries)}")
    print(f"Числа збігаються з таблицею режимів: {'так' if passed else 'НІ'}")


if __name__ == "__main__":
    main()