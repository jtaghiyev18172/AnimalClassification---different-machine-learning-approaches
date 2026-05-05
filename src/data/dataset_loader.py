from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """
    Backward-compatible dataset loader for image classification splits.

    Original supported usage (preserved):
        ImageDataset(split_csv=..., transform=..., classes_to_idx=...)

    Extended usage:
        ImageDataset(
            df=...,
            transform=...,
            classes_to_idx=...,
            project_root=...,
            normalize_paths=True,
            drop_missing=True,
        )

    Notes:
    - By default, path normalization and missing-file dropping are OFF to preserve
      previous behavior exactly.
    - When enabled, path normalization helps make split CSV filepaths portable across
      machines/OS, especially when CSVs contain absolute paths from another device.
    """

    def __init__(
        self,
        split_csv: Optional[str | Path] = None,
        transform: Optional[Callable] = None,
        classes_to_idx: Optional[Dict[str, int]] = None,
        df: Optional[pd.DataFrame] = None,
        project_root: Optional[str | Path] = None,
        normalize_paths: bool = False,
        drop_missing: bool = False,
    ) -> None:
        if split_csv is None and df is None:
            raise ValueError("Either 'split_csv' or 'df' must be provided.")

        if split_csv is not None and df is not None:
            raise ValueError("Provide only one of 'split_csv' or 'df', not both.")

        self.split_csv = Path(split_csv) if split_csv is not None else None
        self.transform = transform
        self.project_root = Path(project_root).expanduser().resolve() if project_root is not None else None
        self.normalize_paths = bool(normalize_paths)
        self.drop_missing = bool(drop_missing)

        if self.split_csv is not None:
            if not self.split_csv.exists():
                raise FileNotFoundError(f"Split CSV not found: {self.split_csv.as_posix()}")
            self.df = pd.read_csv(self.split_csv)
        else:
            self.df = df.copy()  # type: ignore[union-attr]

        required = {"filepath", "label"}
        if not required.issubset(self.df.columns):
            raise ValueError(f"CSV/DataFrame must contain columns {sorted(required)}")

        if self.normalize_paths:
            self.df["filepath"] = self.df["filepath"].apply(self._normalize_filepath)

        if self.drop_missing:
            exists = self.df["filepath"].apply(lambda p: Path(p).exists())
            self.df = self.df[exists].copy()

        self.df = self.df.reset_index(drop=True)

        if classes_to_idx is None:
            labels = sorted(self.df["label"].astype(str).unique().tolist())
            self.classes_to_idx = {c: i for i, c in enumerate(labels)}
        else:
            self.classes_to_idx = dict(classes_to_idx)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        path = Path(row["filepath"])
        label_str = str(row["label"])

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path.as_posix()}")

        with Image.open(path) as im:
            im = im.convert("RGB")

        if self.transform is not None:
            x = self.transform(im)
        else:
            x = torch.from_numpy(
                __import__("numpy").array(im).transpose(2, 0, 1)
            ).float() / 255.0

        y = self.classes_to_idx[label_str]
        return x, y

    def _normalize_filepath(self, p: str) -> str:
        """
        Makes filepaths more portable across machines/OS.

        Handles:
        - Windows absolute paths containing /data/prepared/...
        - already-relative paths starting with data/...
        - backslashes -> slashes

        Behavior:
        - If project_root is not provided, returns the original path unchanged.
        - Only used when normalize_paths=True.
        """
        if self.project_root is None:
            return str(p)

        p = str(p).strip()
        p2 = p.replace("\\", "/")

        marker = "/data/prepared/"
        low = p2.lower()

        if marker in low:
            idx = low.find(marker)
            rel = p2[idx + 1:]  # "data/prepared/..."
            return str((self.project_root / rel).resolve())

        if p2.startswith("data/"):
            return str((self.project_root / p2).resolve())

        original_path = Path(p).expanduser()
        if original_path.exists():
            return str(original_path.resolve())

        return str(p)

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns a copy of the internal dataframe after any optional normalization
        and filtering steps.
        """
        return self.df.copy()