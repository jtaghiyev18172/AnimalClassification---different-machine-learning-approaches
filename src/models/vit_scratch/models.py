from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Literal

import torch
import torch.nn as nn


TrainStage = Literal["head_only", "full_train"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    image_size: int = 224
    resize_size: int = 256
    patch_size: int = 16
    embed_dim: int = 192
    depth: int = 6
    num_heads: int = 3
    mlp_ratio: float = 4.0
    dropout_p: float = 0.1
    attention_dropout_p: float = 0.0
    num_classes: int = 3
    input_channels: int = 3

    def to_dict(self) -> Dict[str, int | float | str]:
        return asdict(self)


SUPPORTED_MODELS: Dict[str, ModelSpec] = {
    "customvit_v1": ModelSpec(
        name="customvit_v1",
        family="vit_scratch",
        image_size=224,
        resize_size=256,
        patch_size=16,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=4.0,
        dropout_p=0.1,
        attention_dropout_p=0.0,
    ),
    "customvit_v2": ModelSpec(
        name="customvit_v2",
        family="vit_scratch",
        image_size=224,
        resize_size=256,
        patch_size=16,
        embed_dim=256,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        dropout_p=0.1,
        attention_dropout_p=0.0,
    ),
}


class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, input_channels: int, embed_dim: int) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size={image_size} must be divisible by patch_size={patch_size}.")

        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(
            input_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input tensor of shape [B, C, H, W], got ndim={x.ndim}.")
        _, _, height, width = x.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"PatchEmbedding expected images of size {self.image_size}x{self.image_size}, "
                f"got {height}x{width}."
            )

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attention_dropout_p: float = 0.0, dropout_p: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}.")

        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3)
        self.attn_dropout = nn.Dropout(p=attention_dropout_p)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, embed_dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, token_count, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attention_scores = (q @ k.transpose(-2, -1)) * self.scale
        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = self.attn_dropout(attention_probs)

        context = attention_probs @ v
        context = context.transpose(1, 2).reshape(batch_size, token_count, embed_dim)
        context = self.proj(context)
        context = self.proj_dropout(context)
        return context


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout_p: float = 0.1) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout_p: float = 0.1, attention_dropout_p: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attention_dropout_p=attention_dropout_p,
            dropout_p=dropout_p,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim=embed_dim, mlp_ratio=mlp_ratio, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerScratch(nn.Module):
    def __init__(self, spec: ModelSpec, num_classes: int = 3, input_channels: int = 3) -> None:
        super().__init__()

        self.spec = spec
        self.image_size = int(spec.image_size)
        self.patch_size = int(spec.patch_size)
        self.embed_dim = int(spec.embed_dim)

        self.patch_embed = PatchEmbedding(
            image_size=self.image_size,
            patch_size=self.patch_size,
            input_channels=input_channels,
            embed_dim=self.embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))
        self.pos_dropout = nn.Dropout(p=spec.dropout_p)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=self.embed_dim,
                    num_heads=spec.num_heads,
                    mlp_ratio=spec.mlp_ratio,
                    dropout_p=spec.dropout_p,
                    attention_dropout_p=spec.attention_dropout_p,
                )
                for _ in range(spec.depth)
            ]
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, num_classes)

        self.apply(initialize_weights_vit)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        batch_size = x.size(0)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_representation = x[:, 0]
        return cls_representation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_representation = self.forward_features(x)
        logits = self.head(cls_representation)
        return logits


class CustomViTv1(VisionTransformerScratch):
    def __init__(self, num_classes: int = 3, input_channels: int = 3) -> None:
        super().__init__(spec=get_model_spec("customvit_v1"), num_classes=num_classes, input_channels=input_channels)


class CustomViTv2(VisionTransformerScratch):
    def __init__(self, num_classes: int = 3, input_channels: int = 3) -> None:
        super().__init__(spec=get_model_spec("customvit_v2"), num_classes=num_classes, input_channels=input_channels)


def initialize_weights_vit(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def get_model_spec(model_name: str) -> ModelSpec:
    name = model_name.strip().lower()
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported scratch ViT model: {model_name}")
    return SUPPORTED_MODELS[name]


def list_available_models() -> Dict[str, str]:
    return {
        "customvit_v1": "Educational ViT-from-scratch baseline",
        "customvit_v2": "Deeper and wider ViT-from-scratch baseline",
    }


def get_recommended_image_size(model_name: str) -> int:
    return int(get_model_spec(model_name).image_size)


def get_recommended_resize_size(model_name: str) -> int:
    return int(get_model_spec(model_name).resize_size)


def build_model(model_name: str, num_classes: int = 3, input_channels: int = 3) -> nn.Module:
    name = model_name.strip().lower()
    if name == "customvit_v1":
        return CustomViTv1(num_classes=num_classes, input_channels=input_channels)
    if name == "customvit_v2":
        return CustomViTv2(num_classes=num_classes, input_channels=input_channels)
    raise ValueError(f"Unsupported model_name: {model_name}")


def _freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def _unfreeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True


def get_head_module(model: nn.Module, model_name: str) -> nn.Module:
    _ = get_model_spec(model_name)
    return model.head


def configure_trainable_stage(model: nn.Module, model_name: str, stage: TrainStage) -> nn.Module:
    _ = get_model_spec(model_name)

    if stage == "head_only":
        _freeze_all(model)
        _unfreeze_module(model.head)
        return model

    if stage == "full_train":
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
) -> List[Dict[str, object]]:
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

    groups: List[Dict[str, object]] = []
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
