from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Literal

import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Small_Weights,
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_V2_S_Weights,
    MobileNet_V3_Large_Weights,
    RegNet_Y_3_2GF_Weights,
    RegNet_Y_8GF_Weights,
    ResNeXt50_32X4D_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    convnext_small,
    densenet121,
    efficientnet_b0,
    efficientnet_b2,
    efficientnet_v2_s,
    mobilenet_v3_large,
    regnet_y_3_2gf,
    regnet_y_8gf,
    resnet18,
    resnet50,
    resnext50_32x4d,
)


TrainStage = Literal["head_only", "partial_finetune", "full_finetune"]
HeadKind = Literal["fc", "classifier", "convnext_classifier"]
PartialStrategy = Literal["module_attr", "children_tail"]


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
    "resnet18_pretrained": ModelSpec(
        name="resnet18_pretrained",
        family="resnet18",
        builder=resnet18,
        default_weights=ResNet18_Weights.DEFAULT,
        head_kind="fc",
        partial_strategy="module_attr",
        partial_source="layer4",
        partial_tail_modules=1,
        image_size=224,
        resize_size=256,
    ),
    "mobilenet_v3_large_pretrained": ModelSpec(
        name="mobilenet_v3_large_pretrained",
        family="mobilenet_v3_large",
        builder=mobilenet_v3_large,
        default_weights=MobileNet_V3_Large_Weights.DEFAULT,
        head_kind="classifier",
        partial_strategy="children_tail",
        partial_source="features",
        partial_tail_modules=2,
        image_size=224,
        resize_size=256,
    ),
    "efficientnet_b0_pretrained": ModelSpec(
        name="efficientnet_b0_pretrained",
        family="efficientnet_b0",
        builder=efficientnet_b0,
        default_weights=EfficientNet_B0_Weights.DEFAULT,
        head_kind="classifier",
        partial_strategy="children_tail",
        partial_source="features",
        partial_tail_modules=2,
        image_size=224,
        resize_size=256,
    ),
    "resnet50_pretrained": ModelSpec(
        name="resnet50_pretrained",
        family="resnet50",
        builder=resnet50,
        default_weights=ResNet50_Weights.DEFAULT,
        head_kind="fc",
        partial_strategy="module_attr",
        partial_source="layer4",
        partial_tail_modules=1,
        image_size=224,
        resize_size=256,
    ),
    "efficientnet_b2_pretrained": ModelSpec(
        name="efficientnet_b2_pretrained",
        family="efficientnet_b2",
        builder=efficientnet_b2,
        default_weights=EfficientNet_B2_Weights.DEFAULT,
        head_kind="classifier",
        partial_strategy="children_tail",
        partial_source="features",
        partial_tail_modules=2,
        image_size=224,
        resize_size=256,
    ),
    "convnext_small": ModelSpec(
        name="convnext_small",
        family="convnext_small",
        builder=convnext_small,
        default_weights=ConvNeXt_Small_Weights.DEFAULT,
        head_kind="convnext_classifier",
        partial_strategy="children_tail",
        partial_source="features",
        partial_tail_modules=2,
        image_size=224,
        resize_size=230,
    ),
    "resnext50_32x4d": ModelSpec(
        name="resnext50_32x4d",
        family="resnext50_32x4d",
        builder=resnext50_32x4d,
        default_weights=ResNeXt50_32X4D_Weights.DEFAULT,
        head_kind="fc",
        partial_strategy="module_attr",
        partial_source="layer4",
        partial_tail_modules=1,
        image_size=224,
        resize_size=232,
    ),
    "densenet121": ModelSpec(
        name="densenet121",
        family="densenet121",
        builder=densenet121,
        default_weights=DenseNet121_Weights.DEFAULT,
        head_kind="classifier",
        partial_strategy="children_tail",
        partial_source="features",
        partial_tail_modules=2,
        image_size=224,
        resize_size=256,
    ),
    "regnet_y_3_2gf": ModelSpec(
        name="regnet_y_3_2gf",
        family="regnet_y_3_2gf",
        builder=regnet_y_3_2gf,
        default_weights=RegNet_Y_3_2GF_Weights.DEFAULT,
        head_kind="fc",
        partial_strategy="children_tail",
        partial_source="trunk_output",
        partial_tail_modules=1,
        image_size=224,
        resize_size=232,
    ),
    "regnet_y_8gf": ModelSpec(
        name="regnet_y_8gf",
        family="regnet_y_8gf",
        builder=regnet_y_8gf,
        default_weights=RegNet_Y_8GF_Weights.DEFAULT,
        head_kind="fc",
        partial_strategy="children_tail",
        partial_source="trunk_output",
        partial_tail_modules=1,
        image_size=224,
        resize_size=232,
    ),
    "efficientnet_v2_s": ModelSpec(
        name="efficientnet_v2_s",
        family="efficientnet_v2_s",
        builder=efficientnet_v2_s,
        default_weights=EfficientNet_V2_S_Weights.DEFAULT,
        head_kind="classifier",
        partial_strategy="children_tail",
        partial_source="features",
        partial_tail_modules=2,
        image_size=384,
        resize_size=384,
    ),
}


def get_model_spec(model_name: str) -> ModelSpec:
    name = model_name.strip().lower()
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported pretrained model: {model_name}")
    return SUPPORTED_MODELS[name]


def list_available_models() -> Dict[str, str]:
    return {
        "resnet18_pretrained": "ResNet18 transfer-learning baseline",
        "mobilenet_v3_large_pretrained": "MobileNetV3-Large transfer-learning baseline",
        "efficientnet_b0_pretrained": "EfficientNet-B0 transfer-learning baseline",
        "resnet50_pretrained": "ResNet50 transfer-learning baseline",
        "efficientnet_b2_pretrained": "EfficientNet-B2 transfer-learning baseline",
        "convnext_small": "ConvNeXt-Small modern ConvNet baseline",
        "resnext50_32x4d": "ResNeXt-50 aggregated residual baseline",
        "densenet121": "DenseNet-121 densely-connected baseline",
        "regnet_y_3_2gf": "RegNetY-3.2GF modern design-space baseline",
        "regnet_y_8gf": "RegNetY-8GF higher-capacity modern baseline",
        "efficientnet_v2_s": "EfficientNetV2-S efficient modern baseline",
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


def _find_first_linear(module: nn.Module) -> nn.Linear:
    if isinstance(module, nn.Linear):
        return module
    if isinstance(module, nn.Sequential):
        for child in module.children():
            if isinstance(child, nn.Linear):
                return child
    raise ValueError(f"Could not find a first nn.Linear inside module type: {type(module).__name__}")


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
    if spec.head_kind == "fc":
        return model.fc
    if spec.head_kind in {"classifier", "convnext_classifier"}:
        return model.classifier
    raise ValueError(f"Unsupported head kind: {spec.head_kind}")


def _replace_convnext_classifier(model: nn.Module, in_features: int, num_classes: int, dropout_p: float) -> nn.Module:
    old_head = model.classifier
    if not isinstance(old_head, nn.Sequential):
        raise ValueError("ConvNeXt classifier replacement expects nn.Sequential classifier.")
    prefix = list(old_head.children())[:-1]
    model.classifier = nn.Sequential(
        *prefix,
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features, num_classes),
    )
    return model


def replace_classifier_head(
    model: nn.Module,
    model_name: str,
    num_classes: int = 3,
    dropout_p: float = 0.3,
) -> nn.Module:
    spec = get_model_spec(model_name)

    if spec.head_kind == "fc":
        in_features = int(model.fc.in_features)
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )
        return model

    if spec.head_kind == "classifier":
        old_head = get_head_module(model, model_name)
        in_features = int(_find_first_linear(old_head).in_features)
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )
        return model

    if spec.head_kind == "convnext_classifier":
        old_head = get_head_module(model, model_name)
        in_features = int(_find_last_linear(old_head).in_features)
        return _replace_convnext_classifier(model, in_features, num_classes, dropout_p)

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

    if spec.partial_strategy == "module_attr":
        return [_resolve_attr_path(model, spec.partial_source)]

    if spec.partial_strategy == "children_tail":
        container = _resolve_attr_path(model, spec.partial_source)
        children = list(container.children())
        if not children:
            raise ValueError(f"Model {model_name} has an empty partial-finetune container at '{spec.partial_source}'.")
        tail = max(1, min(spec.partial_tail_modules, len(children)))
        return children[-tail:]

    raise ValueError(f"Unsupported partial strategy: {spec.partial_strategy}")


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
