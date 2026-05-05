from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.models.cnn_pretrained.utils import (
    EpochMetrics,
    RunResolution,
    TrainingHistory,
    attempt_onnx_export,
    atomic_save_json,
    benchmark_inference,
    build_experiment_signature,
    build_metrics_payload,
    build_training_config,
    collect_device_info,
    count_parameters,
    create_grad_scaler,
    ensure_dir,
    evaluate_model,
    find_runs_by_signature,
    is_run_complete,
    make_run_dir,
    model_size_mb_from_state_dict,
    resolve_run_directory,
    save_checkpoint_atomic,
    save_report_metrics_copy,
    save_training_curves,
    train_one_epoch,
    training_history_from_dict,
)


@dataclass
class ScratchResumeState:
    history: TrainingHistory
    best_state: Dict[str, Any]
    start_epoch: int


def get_group_lr(optimizer: torch.optim.Optimizer, group_name: str) -> float:
    for group in optimizer.param_groups:
        if group.get("name") == group_name:
            return float(group["lr"])
    return float("nan")


def restore_best_weights(model: nn.Module, best_state: Dict[str, Any]) -> nn.Module:
    if not best_state or "model_state_dict" not in best_state:
        raise ValueError("best_state is empty or missing model_state_dict.")
    model.load_state_dict(best_state["model_state_dict"])
    return model


def load_training_resume_state(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    experiment_signature: Optional[str] = None,
    map_location: str = "cpu",
) -> ScratchResumeState:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if experiment_signature is not None and checkpoint.get("experiment_signature") != experiment_signature:
        raise ValueError("Resume checkpoint signature does not match current experiment signature.")

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    history = training_history_from_dict(checkpoint.get("history"))
    best_state = checkpoint.get("best_state", {})
    start_epoch = int(checkpoint.get("completed_epoch", len(history.epochs)))

    return ScratchResumeState(history=history, best_state=best_state, start_epoch=start_epoch)


def fit_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    device: str,
    num_classes: int,
    epochs: int,
    model_name: str,
    checkpoint_path: Path,
    config: Dict[str, Any],
    experiment_signature: str,
    history: Optional[TrainingHistory] = None,
    best_state: Optional[Dict[str, Any]] = None,
    start_epoch: int = 0,
    amp_enabled: bool = False,
    scaler: Optional[torch.amp.GradScaler] = None,
    grad_clip_max_norm: Optional[float] = 1.0,
) -> tuple[TrainingHistory, Dict[str, Any]]:
    if history is None:
        history = TrainingHistory()
    if best_state is None:
        best_state = {}

    best_val_macro_f1 = float(best_state.get("best_val_macro_f1", -math.inf))

    for epoch in range(start_epoch + 1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            amp_enabled=amp_enabled,
            scaler=scaler,
            grad_clip_max_norm=grad_clip_max_norm,
        )

        val_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            amp_enabled=amp_enabled,
        )

        if scheduler is not None:
            scheduler.step(val_metrics["loss"])

        head_lr = get_group_lr(optimizer, "head")
        backbone_lr = get_group_lr(optimizer, "backbone")
        history.epochs.append(
            EpochMetrics(
                epoch=int(epoch),
                stage="full_train",
                train_loss=float(train_loss),
                train_accuracy=float(train_acc),
                val_loss=float(val_metrics["loss"]),
                val_accuracy=float(val_metrics["accuracy"]),
                val_macro_f1=float(val_metrics["macro_f1"]),
                learning_rate_backbone=float(backbone_lr),
                learning_rate_head=float(head_lr),
            )
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            best_state = {
                "epoch": int(epoch),
                "stage": "full_train",
                "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "best_val_macro_f1": best_val_macro_f1,
                "best_val_loss": float(val_metrics["loss"]),
                "best_val_accuracy": float(val_metrics["accuracy"]),
            }
            if scheduler is not None:
                best_state["scheduler_state_dict"] = copy.deepcopy(scheduler.state_dict())
            if scaler is not None:
                best_state["scaler_state_dict"] = copy.deepcopy(scaler.state_dict())

        checkpoint_payload: Dict[str, Any] = {
            "model_name": model_name,
            "experiment_signature": experiment_signature,
            "completed_epoch": int(epoch),
            "config": config,
            "history": history.to_dict(),
            "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "best_state": best_state,
        }
        if scheduler is not None:
            checkpoint_payload["scheduler_state_dict"] = copy.deepcopy(scheduler.state_dict())
        if scaler is not None:
            checkpoint_payload["scaler_state_dict"] = copy.deepcopy(scaler.state_dict())

        save_checkpoint_atomic(checkpoint_path, checkpoint_payload)

        print(
            f"[full_train] "
            f"[Epoch {epoch:02d}/{epochs:02d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f} "
            f"lr_backbone={backbone_lr:.6f} lr_head={head_lr:.6f}"
        )

    return history, best_state
