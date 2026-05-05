from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Literal

import torch.nn as nn
from torchvision.models import (
    MaxVit_T_Weights,
    Swin_T_Weights,
    Swin_V2_S_Weights,
    ViT_B_16_Weights,
    maxvit_t,
    swin_t,
    swin_v2_s,
    vit_b_16,
)


TrainStage = Literal["head_only", "partial_finetune", "full_finetune"]
HeadKind = Literal["head", "heads", "maxvit_classifier"]
PartialStrategy = Literal["children_tail"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    builder: Callable[..., nn.Module]
    default_weights: Any
    head_kind: HeadKind
    partial_strategy: PartialStrategy
    partial_source: str
    partial_tail_modules: int = 1
    image_size: int = 224
    resize_size: int = 256


SUPPORTED_MODELS: Dict[str, ModelSpec] = {
    "vit_b_16": ModelSpec(
        name="vit_b_16",
        family="vit_b_16",
        builder=vit_b_16,
        default_weights=ViT_B_16_Weights.DEFAULT,
        head_kind="heads",
        partial_strategy="children_tail",
        partial_source="encoder.layers",
        partial_tail_modules=2,
        image_size=224,
        resize_size=256,
    ),
    "swin_t": ModelSpec(
        name="swin_t",
        family="swin_t",
        builder=swin_t,
        default_weights=Swin_T_Weights.DEFAULT,
        head_kind="head",
        partial_strategy="children_tail",
        partial_source="features",
        partial_tail_modules=2,
        image_size=224,
        resize_size=232,
    ),
    "swin_v2_s": ModelSpec(
        name="swin_v2_s",
        family="swin_v2_s",
        builder=swin_v2_s,
        default_weights=Swin_V2_S_Weights.DEFAULT,
        head_kind="head",
        partial_strategy="children_tail",
        partial_source="features",
        partial_tail_modules=2,
        image_size=256,
        resize_size=260,
    ),
    "maxvit_t": ModelSpec(
        name="maxvit_t",
        family="maxvit_t",
        builder=maxvit_t,
        default_weights=MaxVit_T_Weights.DEFAULT,
        head_kind="maxvit_classifier",
        partial_strategy="children_tail",
        partial_source="blocks",
        partial_tail_modules=1,
        image_size=224,
        resize_size=224,
    ),
}


def get_model_spec(model_name: str) -> ModelSpec:
    name = model_name.strip().lower()
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported transformer model: {model_name}")
    return SUPPORTED_MODELS[name]


def list_available_models() -> Dict[str, str]:
    return {
        "vit_b_16": "Vision Transformer base patch-16 baseline",
        "swin_t": "Swin Transformer tiny hierarchical baseline",
        "swin_v2_s": "Swin Transformer V2 small hierarchical baseline",
        "maxvit_t": "MaxViT tiny multi-axis attention baseline",
    }


def resolve_weights(model_name: str, pretrained: bool = True) -> Any:
    spec = get_model_spec(model_name)
    return spec.default_weights if pretrained else None


def get_weights_name(model_name: str, pretrained: bool = True) -> str:
    weights = resolve_weights(model_name=model_name, pretrained=pretrained)
    if weights is None:
        return "None"
    return str(getattr(weights, "name", "DEFAULT"))


def get_recommended_image_size(model_name: str) -> int:
    return int(get_model_spec(model_name).image_size)


def get_recommended_resize_size(model_name: str) -> int:
    return int(get_model_spec(model_name).resize_size)


def _resolve_attr_path(module: nn.Module, attr_path: str) -> nn.Module:
    current = module
    for part in attr_path.split("."):
        current = getattr(current, part)
    return current


def _find_last_linear(module: nn.Module) -> nn.Linear:
    if isinstance(module, nn.Linear):
        return module
    if isinstance(module, nn.Sequential):
        for child in reversed(list(module.children())):
            if isinstance(child, nn.Linear):
                return child
    raise ValueError(f"Could not find a last nn.Linear inside module type: {type(module).__name__}")


def get_head_module(model: nn.Module, model_name: str) -> nn.Module:
    spec = get_model_spec(model_name)
    if spec.head_kind == "head":
        return model.head
    if spec.head_kind == "heads":
        return model.heads
    if spec.head_kind == "maxvit_classifier":
        return model.classifier
    raise ValueError(f"Unsupported head kind: {spec.head_kind}")


def _replace_head_attr(model: nn.Module, attr_name: str, num_classes: int, dropout_p: float) -> nn.Module:
    old_head = getattr(model, attr_name)
    in_features = int(old_head.in_features)
    setattr(
        model,
        attr_name,
        nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        ),
    )
    return model


def _replace_last_linear_in_sequential(module: nn.Sequential, num_classes: int, dropout_p: float) -> nn.Sequential:
    children = list(module.children())
    if not children:
        raise ValueError("Expected a non-empty sequential module for head replacement.")
    last_linear = _find_last_linear(module)
    in_features = int(last_linear.in_features)
    prefix = children[:-1]
    return nn.Sequential(
        *prefix,
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features, num_classes),
    )


def replace_classifier_head(
    model: nn.Module,
    model_name: str,
    num_classes: int = 3,
    dropout_p: float = 0.3,
) -> nn.Module:
    spec = get_model_spec(model_name)

    if spec.head_kind == "head":
        return _replace_head_attr(model, "head", num_classes, dropout_p)

    if spec.head_kind == "heads":
        old_head = get_head_module(model, model_name)
        if not isinstance(old_head, nn.Sequential):
            raise ValueError("ViT head replacement expects nn.Sequential heads.")
        model.heads = _replace_last_linear_in_sequential(old_head, num_classes, dropout_p)
        return model

    if spec.head_kind == "maxvit_classifier":
        old_head = get_head_module(model, model_name)
        if not isinstance(old_head, nn.Sequential):
            raise ValueError("MaxViT classifier replacement expects nn.Sequential classifier.")
        model.classifier = _replace_last_linear_in_sequential(old_head, num_classes, dropout_p)
        return model

    raise ValueError(f"Unsupported head kind: {spec.head_kind}")


def build_model(
    model_name: str,
    num_classes: int = 3,
    pretrained: bool = True,
    dropout_p: float = 0.3,
) -> nn.Module:
    spec = get_model_spec(model_name)
    weights = resolve_weights(model_name=model_name, pretrained=pretrained)
    model = spec.builder(weights=weights)
    model = replace_classifier_head(model=model, model_name=model_name, num_classes=num_classes, dropout_p=dropout_p)
    return model


def _freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def _unfreeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True


def _get_partial_backbone_modules(model: nn.Module, model_name: str) -> List[nn.Module]:
    spec = get_model_spec(model_name)
    container = _resolve_attr_path(model, spec.partial_source)
    children = list(container.children())
    if not children:
        raise ValueError(f"Model {model_name} has an empty partial-finetune container at '{spec.partial_source}'.")
    tail = max(1, min(spec.partial_tail_modules, len(children)))
    return children[-tail:]


def configure_trainable_stage(model: nn.Module, model_name: str, stage: TrainStage) -> nn.Module:
    _freeze_all(model)
    _unfreeze_module(get_head_module(model, model_name))

    if stage == "head_only":
        return model

    if stage == "partial_finetune":
        for module in _get_partial_backbone_modules(model, model_name):
            _unfreeze_module(module)
        return model

    if stage == "full_finetune":
        for param in model.parameters():
            param.requires_grad = True
        return model

    raise ValueError(f"Unsupported training stage: {stage}")


def get_trainable_parameter_groups(
    model: nn.Module,
    model_name: str,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> List[Dict[str, Any]]:
    head_ids = {id(p) for p in get_head_module(model, model_name).parameters() if p.requires_grad}
    head_params: List[nn.Parameter] = []
    backbone_params: List[nn.Parameter] = []

    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in head_ids:
            head_params.append(param)
        else:
            backbone_params.append(param)

    groups: List[Dict[str, Any]] = []
    if backbone_params:
        groups.append(
            {
                "params": backbone_params,
                "lr": float(backbone_lr),
                "weight_decay": float(weight_decay),
                "name": "backbone",
            }
        )
    if head_params:
        groups.append(
            {
                "params": head_params,
                "lr": float(head_lr),
                "weight_decay": float(weight_decay),
                "name": "head",
            }
        )
    return groups


def iter_trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for param in model.parameters():
        if param.requires_grad:
            yield param
