from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


def scan_prepared_dataset(prepared_dir: Path, classes: List[str]) -> pd.DataFrame:
    rows: List[Tuple[str, str]] = []

    for c in classes:
        class_dir = prepared_dir / c
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                rows.append((p.as_posix(), c))

    if not rows:
        raise ValueError(f"No images found under: {prepared_dir.as_posix()}")

    df = pd.DataFrame(rows, columns=["filepath", "label"])
    return df


def make_stratified_splits(
    df: pd.DataFrame,
    cfg: SplitConfig,
) -> Dict[str, pd.DataFrame]:
    ratios_sum = cfg.train_ratio + cfg.val_ratio + cfg.test_ratio
    if not np.isclose(ratios_sum, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios_sum}")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - cfg.train_ratio),
        random_state=cfg.seed,
        stratify=df["label"],
    )

    val_fraction_of_temp = cfg.val_ratio / (cfg.val_ratio + cfg.test_ratio)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_fraction_of_temp),
        random_state=cfg.seed,
        stratify=temp_df["label"],
    )

    train_df = train_df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)

    return {"train": train_df, "val": val_df, "test": test_df}


def write_splits(
    splits: Dict[str, pd.DataFrame],
    out_dir: Path,
    classes_to_idx: Dict[str, int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    splits["train"].to_csv(out_dir / "train.csv", index=False)
    splits["val"].to_csv(out_dir / "val.csv", index=False)
    splits["test"].to_csv(out_dir / "test.csv", index=False)

    (out_dir / "classes.json").write_text(
        pd.Series(classes_to_idx).to_json(),
        encoding="utf-8",
    )


def split_stats(df: pd.DataFrame) -> Dict[str, int]:
    return df["label"].value_counts().to_dict()


def validate_splits(
    original_df: pd.DataFrame,
    splits: Dict[str, pd.DataFrame],
    classes: List[str],
) -> Dict[str, Dict[str, float]]:
    original_counts = original_df["label"].value_counts().reindex(classes, fill_value=0)
    original_total = int(original_counts.sum())
    original_ratios = (original_counts / max(original_total, 1)).to_dict()

    out: Dict[str, Dict[str, float]] = {"original": {k: float(v) for k, v in original_ratios.items()}}

    for name, sdf in splits.items():
        counts = sdf["label"].value_counts().reindex(classes, fill_value=0)
        total = int(counts.sum())
        ratios = (counts / max(total, 1)).to_dict()
        out[name] = {k: float(v) for k, v in ratios.items()}

    return out