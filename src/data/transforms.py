from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from torchvision import transforms


def load_transforms_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Transforms config not found: {p.as_posix()}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Invalid transforms config: expected a YAML mapping at the top level.")
    return data


def _get_normalize(cfg: Dict[str, Any]) -> transforms.Normalize:
    norm = cfg.get("normalize", None)
    if not isinstance(norm, dict):
        raise ValueError("Missing 'normalize' section in transforms config.")
    mean = norm.get("mean", None)
    std = norm.get("std", None)
    if mean is None or std is None:
        raise ValueError("Normalize config must include 'mean' and 'std'.")
    return transforms.Normalize(mean=mean, std=std)


def _build_from_spec_list(
    cfg: Dict[str, Any],
    spec_list: List[Dict[str, Any]],
) -> transforms.Compose:
    ops = []
    normalize_op = _get_normalize(cfg)

    for item in spec_list:
        if not isinstance(item, dict) or "name" not in item:
            raise ValueError("Each transform spec must be a dict with a 'name' key.")
        name = str(item["name"]).lower()
        params = item.get("params", {}) or {}
        if not isinstance(params, dict):
            raise ValueError(f"Transform params must be a dict for '{name}'.")

        if name == "random_resized_crop":
            size = params.get("size", cfg.get("image_size", 224))
            scale = tuple(params.get("scale", (0.7, 1.0)))
            ratio = tuple(params.get("ratio", (0.75, 1.3333)))
            ops.append(transforms.RandomResizedCrop(size=size, scale=scale, ratio=ratio))

        elif name == "random_horizontal_flip":
            p = float(params.get("p", 0.5))
            ops.append(transforms.RandomHorizontalFlip(p=p))

        elif name == "random_rotation":
            degrees = params.get("degrees", 15)
            ops.append(transforms.RandomRotation(degrees=degrees))

        elif name == "color_jitter":
            ops.append(
                transforms.ColorJitter(
                    brightness=params.get("brightness", 0.2),
                    contrast=params.get("contrast", 0.2),
                    saturation=params.get("saturation", 0.2),
                    hue=params.get("hue", 0.05),
                )
            )

        elif name == "resize":
            size = params.get("size", cfg.get("resize_size", 256))
            ops.append(transforms.Resize(size=size))

        elif name == "center_crop":
            size = params.get("size", cfg.get("image_size", 224))
            ops.append(transforms.CenterCrop(size=size))

        elif name == "to_tensor":
            ops.append(transforms.ToTensor())

        elif name == "normalize":
            ops.append(normalize_op)

        else:
            raise ValueError(f"Unknown transform name: '{name}'")

    return transforms.Compose(ops)


def get_train_transforms(config: Dict[str, Any]) -> transforms.Compose:
    spec_list = config.get("train_transforms", None)
    if not isinstance(spec_list, list):
        raise ValueError("Missing or invalid 'train_transforms' list in config.")
    return _build_from_spec_list(config, spec_list)


def get_eval_transforms(config: Dict[str, Any]) -> transforms.Compose:
    spec_list = config.get("eval_transforms", None)
    if not isinstance(spec_list, list):
        raise ValueError("Missing or invalid 'eval_transforms' list in config.")
    return _build_from_spec_list(config, spec_list)


def apply_size_overrides(
    config: Dict[str, Any],
    image_size: Optional[int] = None,
    resize_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Return a deep-copied transform config with runtime size overrides applied.

    This keeps the base YAML config reusable while allowing model-family notebooks
    to safely align training/eval crop sizes with backbone-specific pretrained
    weight expectations.
    """
    cfg = copy.deepcopy(config)

    if image_size is not None:
        cfg["image_size"] = int(image_size)
    if resize_size is not None:
        cfg["resize_size"] = int(resize_size)

    for item in cfg.get("train_transforms", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).lower()
        params = item.setdefault("params", {})
        if name == "random_resized_crop" and image_size is not None:
            params["size"] = int(image_size)

    for item in cfg.get("eval_transforms", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).lower()
        params = item.setdefault("params", {})
        if name == "resize" and resize_size is not None:
            params["size"] = int(resize_size)
        elif name == "center_crop" and image_size is not None:
            params["size"] = int(image_size)

    return cfg
