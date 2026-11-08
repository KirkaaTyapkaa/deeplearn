"""ЛР3. Команда навчання: поділ, перевірки пунктів 1-2, сім запусків, журнали, вибір за validation.

    uv run python train.py --data /шлях/до/набору [--runs runs] [--model lab3_model.pt] [--out outputs]

Порядок роботи:
[1] поділ -> split.csv і контрольна таблиця; перевірки пунктів 1-2 (checks.py);
    якщо хоч одна перевірка не зійшлася, навчання не запускається;
[2] сім запусків за протоколом: runs/<режим>_seed<зерно>/run.json, val_logits.npy, model.pt;
[3] вибір конфігурації лише за validation і збереження lab3_model.pt.

Тестова вибірка тут не завантажується. Тека --runs має бути порожньою або не існувати:
результати, за якими складено звіт, лежать у runs/, тому повторне навчання йде в іншу теку.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

import data
from checks import run_checks
from data import CLASSES, IMG_SIZE, SPLIT_PATH, build_split, control_table, load_split, make_loader, save_split, split_is_valid
from model import REGIMES, WEIGHT_DECAY, Regime, build_model, pick_device, set_train_mode

EPOCHS = 8
# Сім запусків протоколу в порядку таблиці умови.
RUNS: tuple[tuple[str, int], ...] = (
    ("scratch", 1),
    ("frozen", 1), ("frozen", 2), ("frozen", 3),
    ("unfreeze", 1), ("unfreeze", 2), ("unfreeze", 3),
)


def run_name(regime: str, seed: int) -> str:
    return f"{regime}_seed{seed}"


# ---------------------------------------------------------------- оточення запуску


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> tuple[str | None, bool | None]:
    """Хеш HEAD і чи є незакомічені зміни у відстежуваних файлах поточної теки."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", "."],
                                capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return head, bool(status.strip())


def environment(device: torch.device) -> dict[str, object]:
    """Усе, що однакове для семи запусків: версія коду й даних, бібліотеки, пристрій."""
    commit, dirty = git_commit()
    return {
        "git_commit": commit,
        "git_uncommitted_changes": dirty,
        "split_sha256": sha256_of(SPLIT_PATH),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_num_threads": torch.get_num_threads(),
        "platform": platform.platform(),
    }


def settings(regime: Regime, seed: int, optimizer: torch.optim.Optimizer, model: nn.Module,
             train_loader: DataLoader) -> dict[str, object]:
    """Усі налаштування запуску, включно з тими, що лишилися за замовчуванням."""
    defaults = {k: (list(v) if isinstance(v, tuple) else v) for k, v in optimizer.defaults.items()}
    return {
        "regime": regime.name,
        "seed": seed,
        "weights": "ResNet18_Weights.IMAGENET1K_V1" if regime.pretrained else None,
        "head": repr(model.fc),
        "trainable_modules": list(regime.trainable),
        "eval_modules_during_training": list(regime.frozen),
        "optimizer": type(optimizer).__name__,
        "optimizer_groups": [{"modules": list(modules), "lr": lr} for modules, lr in regime.groups],
        "optimizer_defaults": defaults,
        "weight_decay": WEIGHT_DECAY,
        "bn_momentum": model.bn1.momentum,
        "bn_eps": model.bn1.eps,
        "loss": "CrossEntropyLoss(reduction='mean')",
        "epochs": EPOCHS,
        "steps_per_epoch": len(train_loader),
        "total_steps": EPOCHS * len(train_loader),
        "img_size": IMG_SIZE,
        "train_transform": repr(data.TRAIN_TRANSFORM),
        "eval_transform": repr(data.EVAL_TRANSFORM),
        "train_loader": {"batch_size": data.BATCH_TRAIN, "shuffle": True, "drop_last": False, "num_workers": 0},
        "eval_loader": {"batch_size": data.BATCH_EVAL, "shuffle": False, "num_workers": 0},
    }


# ---------------------------------------------------------------- один запуск


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Логіти й мітки в порядку завантажувача; model.eval(), без градієнтів."""
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            logits.append(model(images.to(device)).cpu())
            labels.append(targets)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def macro_f1(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(f1_score(labels, logits.argmax(axis=1), average="macro"))


def train_run(regime: Regime, seed: int, root: Path, split, device: torch.device,
              run_dir: Path, env: dict[str, object]) -> dict[str, object]:
    """Вісім епох за протоколом; журнал, логіти validation після 8-ї епохи і ваги зберігаються в run_dir."""
    model, optimizer = build_model(regime, seed, device)
    train_loader = make_loader(root, split, "train")
    val_loader = make_loader(root, split, "validation")
    loss_fn = nn.CrossEntropyLoss()
    n_train = len(train_loader.dataset)

    train_loss: list[float] = []
    val_f1: list[float] = []
    train_time = 0.0
    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()
        set_train_mode(model, regime)
        loss_sum = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(targets)  # втрата пакета до кроку, зважена його розміром
        train_time += time.perf_counter() - started

        logits, labels = evaluate(model, val_loader, device)
        train_loss.append(loss_sum / n_train)
        val_f1.append(macro_f1(logits, labels))
        print(f"    епоха {epoch}: втрата train {train_loss[-1]:.4f}, macro-F1 validation {val_f1[-1]:.4f}")

    run_dir.mkdir(parents=True, exist_ok=True)
    log = {**settings(regime, seed, optimizer, model, train_loader), **env,
           "train_loss": train_loss, "val_macro_f1": val_f1, "train_time_s": train_time}
    (run_dir / "run.json").write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    np.save(run_dir / "val_logits.npy", logits)
    torch.save(model.state_dict(), run_dir / "model.pt")
    return log


# ---------------------------------------------------------------- вибір


def choose(logs: list[dict[str, object]]) -> tuple[str, int]:
    """Режим із найбільшим середнім macro-F1 за зернами, у ньому запуск із найбільшим macro-F1."""
    by_regime: dict[str, list[dict[str, object]]] = {}
    for log in logs:
        by_regime.setdefault(log["regime"], []).append(log)
    best_regime = max(by_regime, key=lambda r: np.mean([log["val_macro_f1"][-1] for log in by_regime[r]]))
    best_run = max(by_regime[best_regime], key=lambda log: log["val_macro_f1"][-1])
    return best_regime, best_run["seed"]


def save_selected(regime: str, seed: int, runs: Path, path: Path) -> None:
    state_dict = torch.load(runs / run_name(regime, seed) / "model.pt", map_location="cpu")
    torch.save({"state_dict": state_dict, "regime": regime, "seed": seed,
                "classes": list(CLASSES), "img_size": IMG_SIZE}, path)


def runs_markdown(logs: list[dict[str, object]]) -> str:
    lines = ["| # | Режим | Зерно | Втрата train, епоха 8 | macro-F1 validation, епоха 8 | Час навчання, с |",
             "|---|---|---|---|---|---|"]
    for i, log in enumerate(logs, 1):
        lines.append(f"| {i} | {log['regime']} | {log['seed']} | {log['train_loss'][-1]:.4f} "
                     f"| {log['val_macro_f1'][-1]:.4f} | {log['train_time_s']:.1f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- команда


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="тека з seg_train/ і seg_test/")
    parser.add_argument("--runs", type=Path, default=Path("runs"), help="тека журналів запусків")
    parser.add_argument("--model", type=Path, default=Path("lab3_model.pt"), help="файл вибраної моделі")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="тека для рисунка пакета і checks.md")
    args = parser.parse_args()

    if args.runs.exists() and any(args.runs.iterdir()):
        sys.exit(f"Тека {args.runs} не порожня. Для повторного навчання вкажіть іншу: --runs runs_rerun.")

    device = pick_device()
    print(f"Пристрій: {device}\n")

    print("[1] Поділ і перевірки до навчання")
    split = build_split(args.data)
    save_split(split)
    split = load_split()
    if not split_is_valid(control_table(split), split):
        sys.exit("Поділ не збігається з контрольною таблицею умови.")
    print("Поділ збігається з контрольною таблицею: так\n")
    if not run_checks(args.data, args.out, device):
        sys.exit("Перевірки пунктів 1-2 не збіглися з очікуваним, навчання не запускається.")

    env = environment(device)
    print(f"\nКоміт {env['git_commit']}, незакомічені зміни: {env['git_uncommitted_changes']}, "
          f"SHA-256 split.csv: {env['split_sha256']}\n")

    print("[2] Сім запусків")
    logs = []
    for i, (regime, seed) in enumerate(RUNS, 1):
        print(f"Запуск {i}: {regime}, зерно {seed}")
        logs.append(train_run(REGIMES[regime], seed, args.data, split, device, args.runs / run_name(regime, seed), env))
        print(f"    час навчання {logs[-1]['train_time_s']:.1f} с")
    print()
    print(runs_markdown(logs))

    print("\n[3] Вибір за validation")
    regime, seed = choose(logs)
    save_selected(regime, seed, args.runs, args.model)
    print(f"Обрано {regime}, зерно {seed}. Ваги збережено в {args.model}")


if __name__ == "__main__":
    main()