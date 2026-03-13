from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float
    learning_rate: float


@dataclass
class TrainingHistory:
    epochs: List[EpochMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"epochs": [asdict(x) for x in self.epochs]}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_run_dir(model_root: Path, prefix: str = "run") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = model_root / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def atomic_save_json(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def save_checkpoint_atomic(path: Path, checkpoint: Dict[str, Any]) -> None:
    tmp = path.with_name(path.stem + "_tmp.pt")
    torch.save(checkpoint, tmp)
    tmp.replace(path)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb_from_state_dict(model: nn.Module) -> float:
    total_bytes = 0
    for tensor in model.state_dict().values():
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024 ** 2)


def top1_accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / max(1, targets.size(0))


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """
    sklearn-free macro F1 implementation.
    """
    f1s: List[float] = []
    for cls_idx in range(num_classes):
        tp = np.logical_and(y_pred == cls_idx, y_true == cls_idx).sum()
        fp = np.logical_and(y_pred == cls_idx, y_true != cls_idx).sum()
        fn = np.logical_and(y_pred != cls_idx, y_true == cls_idx).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        f1s.append(f1)

    return float(np.mean(f1s))


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip_max_norm: Optional[float] = 1.0,
) -> Tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()

        if grad_clip_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)

        optimizer.step()

        batch_size = yb.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())
        total_seen += int(batch_size)

    avg_loss = total_loss / max(1, total_seen)
    avg_acc = total_correct / max(1, total_seen)
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
    num_classes: int,
) -> Dict[str, Any]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = criterion(logits, yb)

        preds = logits.argmax(dim=1)

        batch_size = yb.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((preds == yb).sum().item())
        total_seen += int(batch_size)

        y_true_all.append(yb.detach().cpu().numpy())
        y_pred_all.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=np.int64)
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([], dtype=np.int64)

    avg_loss = total_loss / max(1, total_seen)
    avg_acc = total_correct / max(1, total_seen)
    macro_f1 = macro_f1_score(y_true, y_pred, num_classes=num_classes) if total_seen > 0 else 0.0
    cm = confusion_matrix_np(y_true, y_pred, num_classes=num_classes) if total_seen > 0 else np.zeros((num_classes, num_classes), dtype=np.int64)

    return {
        "loss": float(avg_loss),
        "accuracy": float(avg_acc),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
        "y_true": y_true,
        "y_pred": y_pred,
    }


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
    grad_clip_max_norm: Optional[float] = 1.0,
) -> Tuple[TrainingHistory, Dict[str, Any]]:
    history = TrainingHistory()
    best_state: Dict[str, Any] = {}
    best_val_macro_f1 = -math.inf

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip_max_norm=grad_clip_max_norm,
        )

        val_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )

        if scheduler is not None:
            scheduler.step(val_metrics["loss"])

        current_lr = get_current_lr(optimizer)

        epoch_metrics = EpochMetrics(
            epoch=epoch,
            train_loss=float(train_loss),
            train_accuracy=float(train_acc),
            val_loss=float(val_metrics["loss"]),
            val_accuracy=float(val_metrics["accuracy"]),
            val_macro_f1=float(val_metrics["macro_f1"]),
            learning_rate=float(current_lr),
        )
        history.epochs.append(epoch_metrics)

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_macro_f1": best_val_macro_f1,
                "best_val_loss": float(val_metrics["loss"]),
                "best_val_accuracy": float(val_metrics["accuracy"]),
            }

        print(
            f"[Epoch {epoch:02d}/{epochs:02d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f} lr={current_lr:.6f}"
        )

    return history, best_state


def restore_best_weights(model: nn.Module, best_state: Dict[str, Any]) -> nn.Module:
    if not best_state or "model_state_dict" not in best_state:
        raise ValueError("best_state is empty or missing model_state_dict.")
    model.load_state_dict(best_state["model_state_dict"])
    return model


@torch.no_grad()
def benchmark_inference(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    warmup_batches: int = 5,
    timed_batches: int = 20,
) -> Dict[str, float]:
    model.eval()

    batches = []
    for xb, _ in loader:
        batches.append(xb)
        if len(batches) >= max(warmup_batches, timed_batches):
            break

    if not batches:
        return {
            "latency_ms_per_batch": float("nan"),
            "latency_ms_per_image": float("nan"),
            "throughput_img_per_sec": float("nan"),
            "num_timed_batches": 0.0,
        }

    for xb in batches[:warmup_batches]:
        xb = xb.to(device, non_blocking=True)
        _ = model(xb)
        if device == "cuda":
            torch.cuda.synchronize()

    timed = batches[:timed_batches]
    total_images = 0
    t0 = time.perf_counter()

    for xb in timed:
        xb = xb.to(device, non_blocking=True)
        _ = model(xb)
        total_images += int(xb.size(0))
        if device == "cuda":
            torch.cuda.synchronize()

    dt = time.perf_counter() - t0
    num_batches = len(timed)

    latency_batch_ms = (dt / max(1, num_batches)) * 1000.0
    latency_img_ms = (dt / max(1, total_images)) * 1000.0
    throughput = total_images / dt if dt > 0 else float("inf")

    return {
        "latency_ms_per_batch": float(latency_batch_ms),
        "latency_ms_per_image": float(latency_img_ms),
        "throughput_img_per_sec": float(throughput),
        "num_timed_batches": float(num_batches),
    }


def export_model_to_onnx(
    model: nn.Module,
    export_path: Path,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = "cpu",
    opset_version: int = 17,
) -> None:
    model.eval()

    dummy_input = torch.randn(*input_shape, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        str(export_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )


def save_training_curves(history: TrainingHistory, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = [e.epoch for e in history.epochs]
    train_loss = [e.train_loss for e in history.epochs]
    val_loss = [e.val_loss for e in history.epochs]
    train_acc = [e.train_accuracy for e in history.epochs]
    val_acc = [e.val_accuracy for e in history.epochs]

    loss_curve_path = output_dir / "loss_curve.png"
    acc_curve_path = output_dir / "accuracy_curve.png"

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_curve_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, label="train_accuracy")
    plt.plot(epochs, val_acc, label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_curve_path, dpi=150)
    plt.close()

    return {
        "loss_curve": loss_curve_path,
        "accuracy_curve": acc_curve_path,
    }


def build_training_config(
    model_name: str,
    split_id: str,
    transform_id_train: str,
    transform_id_eval: str,
    dataset_version: str,
    training_params: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = {
        "model_name": model_name,
        "split_id": split_id,
        "transform_id_train": transform_id_train,
        "transform_id_eval": transform_id_eval,
        "dataset_version": dataset_version,
        "training_params": training_params,
        "timestamp": datetime.now().isoformat(),
    }
    if extra:
        cfg.update(extra)
    return cfg


def build_metrics_payload(
    history: TrainingHistory,
    best_state: Dict[str, Any],
    test_metrics: Dict[str, Any],
    benchmark_metrics: Dict[str, float],
    parameter_count: int,
    model_size_mb: float,
) -> Dict[str, Any]:
    return {
        "history": history.to_dict(),
        "best_epoch": int(best_state.get("epoch", -1)),
        "best_val_macro_f1": float(best_state.get("best_val_macro_f1", float("nan"))),
        "best_val_loss": float(best_state.get("best_val_loss", float("nan"))),
        "best_val_accuracy": float(best_state.get("best_val_accuracy", float("nan"))),
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_confusion_matrix": test_metrics["confusion_matrix"],
        "latency_ms_per_batch": float(benchmark_metrics["latency_ms_per_batch"]),
        "latency_ms_per_image": float(benchmark_metrics["latency_ms_per_image"]),
        "throughput_img_per_sec": float(benchmark_metrics["throughput_img_per_sec"]),
        "parameter_count": int(parameter_count),
        "model_size_mb": float(model_size_mb),
    }