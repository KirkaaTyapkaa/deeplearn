"""ЛР3. Дані: поділ за процедурою умови, контрольна таблиця, набір даних і завантажувачі.

Окремий запуск будує split.csv у поточній теці й друкує контрольну таблицю:
    uv run python data.py --data /шлях/до/набору

--data вказує на теку, що містить seg_train/ і seg_test/. Тека seg_pred не використовується.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASSES = ("buildings", "forest", "glacier", "mountain", "sea", "street")
PARTS = ("train", "validation", "test")
TRAIN_DIR = Path("seg_train", "seg_train")
TEST_DIR = Path("seg_test", "seg_test")
PART_DIR = {"train": TRAIN_DIR, "validation": TEST_DIR, "test": TEST_DIR}

TRAIN_PER_CLASS = 200
SPLIT_SEED = 0
IMG_SIZE = 160
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BATCH_TRAIN = 32
BATCH_EVAL = 64
SPLIT_PATH = Path("split.csv")

# Контрольна таблиця з умови: (кількість, сума числових імен) для train, validation, test.
CONTROL: dict[str, tuple[tuple[int, int], ...]] = {
    "buildings": ((200, 1_996_761), (218, 4_837_323), (219, 4_837_559)),
    "forest": ((200, 2_032_625), (237, 5_276_962), (237, 5_291_326)),
    "glacier": ((200, 2_183_923), (276, 6_135_652), (277, 6_157_250)),
    "mountain": ((200, 1_950_345), (262, 5_789_535), (263, 5_849_765)),
    "sea": ((200, 1_987_613), (255, 5_655_068), (255, 5_668_011)),
    "street": ((200, 1_841_817), (250, 5_541_845), (251, 5_547_533)),
}

# Аугментація лише в train; validation і test детерміновані. Нормування ImageNet в усіх режимах.
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ---------------------------------------------------------------- поділ


def class_names(directory: Path) -> list[str]:
    """Імена файлів одного класу, відсортовані як рядки: 0.jpg, 10006.jpg, 1001.jpg, ..."""
    return sorted(p.name for p in directory.iterdir() if p.suffix == ".jpg")


def shuffled(rng: np.random.Generator, n: int) -> np.ndarray:
    idx = np.arange(n)
    rng.shuffle(idx)
    return idx


def build_split(root: Path) -> pd.DataFrame:
    """Процедура умови: один генератор на всі класи по черзі для train, новий - для validation і test."""
    rows: list[tuple[str, str, str]] = []

    rng = np.random.default_rng(SPLIT_SEED)
    for label in CLASSES:
        names = class_names(root / TRAIN_DIR / label)
        idx = shuffled(rng, len(names))
        rows += [(names[i], label, "train") for i in idx[:TRAIN_PER_CLASS]]

    rng = np.random.default_rng(SPLIT_SEED)
    for label in CLASSES:
        names = class_names(root / TEST_DIR / label)
        idx = shuffled(rng, len(names))
        cut = len(names) // 2
        rows += [(names[i], label, "validation") for i in idx[:cut]]
        rows += [(names[i], label, "test") for i in idx[cut:]]

    return pd.DataFrame(rows, columns=["file", "label", "part"])


def save_split(split: pd.DataFrame, path: Path = SPLIT_PATH) -> None:
    """Однаковий кінець рядка на всіх платформах, щоб SHA-256 файла не залежав від ОС."""
    split.to_csv(path, index=False, lineterminator="\n")


def load_split(path: Path = SPLIT_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------------------------------------------------------- контроль


def control_table(split: pd.DataFrame) -> pd.DataFrame:
    """Для кожного класу й частини: кількість, сума числових імен і збіг з таблицею умови."""
    numbers = split["file"].str.removesuffix(".jpg").astype(int)
    grouped = split.assign(number=numbers).groupby(["label", "part"])["number"].agg(["count", "sum"])
    rows = []
    for label in CLASSES:
        row: dict[str, object] = {"class": label}
        for part, expected in zip(PARTS, CONTROL[label], strict=True):
            count, total = (int(v) for v in grouped.loc[(label, part)])
            row[f"{part}_count"] = count
            row[f"{part}_sum"] = total
            row[f"{part}_match"] = (count, total) == expected
        rows.append(row)
    return pd.DataFrame(rows)


def split_is_valid(table: pd.DataFrame, split: pd.DataFrame) -> bool:
    """Усі 18 клітинок збігаються і жоден файл не входить у дві частини."""
    cells_match = bool(table[[f"{part}_match" for part in PARTS]].all().all())
    unique_files = not split["file"].duplicated().any()
    return cells_match and unique_files


def control_markdown(table: pd.DataFrame) -> str:
    """Таблиця контролю у форматі умови: «кількість / сума» у кожній клітинці."""
    lines = ["| Клас | train | validation | test | Збіг |", "|---|---|---|---|---|"]
    for _, r in table.iterrows():
        cells = [f"{r[f'{p}_count']} / {r[f'{p}_sum']:,}".replace(",", " ") for p in PARTS]
        ok = all(r[f"{p}_match"] for p in PARTS)
        lines.append(f"| {r['class']} | " + " | ".join(cells) + f" | {'так' if ok else 'ні'} |")
    totals = [
        f"{table[f'{p}_count'].sum()} / {table[f'{p}_sum'].sum():,}".replace(",", " ") for p in PARTS
    ]
    lines.append("| Разом | " + " | ".join(totals) + " | |")
    return "\n".join(lines)


# ---------------------------------------------------------------- набір даних


class IntelImages(Dataset):
    """Зображення однієї частини поділу у порядку рядків split.csv."""

    def __init__(self, root: Path, split: pd.DataFrame, part: str, transform: transforms.Compose):
        rows = split[split["part"] == part]
        self.paths = [root / PART_DIR[part] / label / file for file, label in zip(rows["file"], rows["label"])]
        self.targets = [CLASSES.index(label) for label in rows["label"]]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        image = Image.open(self.paths[i]).convert("RGB")
        return self.transform(image), self.targets[i]


def make_loader(root: Path, split: pd.DataFrame, part: str) -> DataLoader:
    """train: 32, shuffle з глобального генератора PyTorch; validation і test: 64, без перемішування."""
    if part == "train":
        dataset = IntelImages(root, split, part, TRAIN_TRANSFORM)
        return DataLoader(dataset, batch_size=BATCH_TRAIN, shuffle=True, drop_last=False, num_workers=0)
    dataset = IntelImages(root, split, part, EVAL_TRANSFORM)
    return DataLoader(dataset, batch_size=BATCH_EVAL, shuffle=False, num_workers=0)


# ---------------------------------------------------------------- запуск


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="тека з seg_train/ і seg_test/")
    args = parser.parse_args()

    split = build_split(args.data)
    save_split(split)
    split = load_split()
    table = control_table(split)

    print(f"split.csv: {len(split)} рядків\n")
    print(control_markdown(table))
    print(f"\nПовторів імен файлів: {int(split['file'].duplicated().sum())}")
    print(f"Поділ збігається з контрольною таблицею умови: {'так' if split_is_valid(table, split) else 'НІ'}")


if __name__ == "__main__":
    main()