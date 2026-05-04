from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


@dataclass
class EpochMetrics:
    epoch: int
    stage: str
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float
    learning_rate_backbone: float
    learning_rate_head: float


@dataclass
class TrainingHistory:
    epochs: List[EpochMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"epochs": [asdict(x) for x in self.epochs]}


@dataclass
class RunResolution:
    action: str
    run_dir: Path
    completed_matches: List[Path] = field(default_factory=list)
    incomplete_matches: List[Path] = field(default_factory=list)
    resumable_checkpoint: Optional[Path] = None


@dataclass
class ResumeState:
    history: TrainingHistory
    best_state: Dict[str, Any]
    completed_stage: Optional[str] = None


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


def read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def model_size_mb_from_state_dict(model: nn.Module) -> float:
    total_bytes = 0
    for tensor in model.state_dict().values():
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024 ** 2)


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1s: List[float] = []
    for cls_idx in range(num_classes):
        tp = np.logical_and(y_pred == cls_idx, y_true == cls_idx).sum()
        fp = np.logical_and(y_pred == cls_idx, y_true != cls_idx).sum()
        fn = np.logical_and(y_pred != cls_idx, y_true == cls_idx).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        f1s.append(f1)
    return float(np.mean(f1s))


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def build_experiment_signature(signature_payload: Dict[str, Any]) -> str:
    canonical = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def build_training_config(
    model_name: str,
    backbone_name: str,
    weights_name: str,
    split_id: str,
    transform_id_train: str,
    transform_id_eval: str,
    dataset_version: str,
    training_params: Dict[str, Any],
    experiment_signature: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = {
        "model_name": model_name,
        "backbone_name": backbone_name,
        "weights_name": weights_name,
        "split_id": split_id,
        "transform_id_train": transform_id_train,
        "transform_id_eval": transform_id_eval,
        "dataset_version": dataset_version,
        "training_params": training_params,
        "experiment_signature": experiment_signature,
        "timestamp": datetime.now().isoformat(),
    }
    if extra:
        cfg.update(extra)
    return cfg


def is_run_complete(run_dir: Path, required_files: Sequence[str] = ("config.json", "metrics.json", "checkpoint.pt")) -> bool:
    return all((run_dir / name).exists() for name in required_files)


def discover_runs(model_root: Path) -> List[Path]:
    if not model_root.exists():
        return []
    return sorted([p for p in model_root.iterdir() if p.is_dir() and p.name.startswith("run_")], key=lambda p: p.name)


def find_runs_by_signature(model_root: Path, experiment_signature: str) -> Tuple[List[Path], List[Path]]:
    completed: List[Path] = []
    incomplete: List[Path] = []
    for run_dir in discover_runs(model_root):
        cfg = read_json_if_exists(run_dir / "config.json")
        if cfg is None:
            continue
        if cfg.get("experiment_signature") != experiment_signature:
            continue
        if is_run_complete(run_dir):
            completed.append(run_dir)
        else:
            incomplete.append(run_dir)
    return completed, incomplete


def resolve_run_directory(
    model_root: Path,
    experiment_signature: str,
    allow_resume: bool = True,
) -> RunResolution:
    ensure_dir(model_root)
    completed, incomplete = find_runs_by_signature(model_root=model_root, experiment_signature=experiment_signature)

    if completed:
        return RunResolution(
            action="reuse",
            run_dir=completed[-1],
            completed_matches=completed,
            incomplete_matches=incomplete,
            resumable_checkpoint=completed[-1] / "checkpoint.pt",
        )

    if allow_resume:
        for run_dir in reversed(incomplete):
            candidate = run_dir / "checkpoint.pt"
            if candidate.exists():
                return RunResolution(
                    action="resume",
                    run_dir=run_dir,
                    completed_matches=completed,
                    incomplete_matches=incomplete,
                    resumable_checkpoint=candidate,
                )

    if incomplete:
        latest_incomplete = incomplete[-1]
        candidate = latest_incomplete / "checkpoint.pt"
        return RunResolution(
            action="restart_incomplete",
            run_dir=latest_incomplete,
            completed_matches=completed,
            incomplete_matches=incomplete,
            resumable_checkpoint=candidate if candidate.exists() else None,
        )

    run_dir = make_run_dir(model_root=model_root, prefix="run")
    return RunResolution(
        action="create",
        run_dir=run_dir,
        completed_matches=completed,
        incomplete_matches=incomplete,
        resumable_checkpoint=None,
    )


def get_group_lr(optimizer: torch.optim.Optimizer, group_name: str) -> float:
    for group in optimizer.param_groups:
        if group.get("name") == group_name:
            return float(group["lr"])
    return float("nan")


def create_grad_scaler(device: str, amp_enabled: bool) -> Optional[torch.amp.GradScaler]:
    if device != "cuda" or not amp_enabled:
        return None
    return torch.amp.GradScaler("cuda")


def collect_device_info(device: str, amp_enabled: bool, num_workers: int, batch_size: int) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "amp_enabled": bool(amp_enabled and device == "cuda"),
        "num_workers": int(num_workers),
        "batch_size": int(batch_size),
    }
    if device == "cuda" and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "gpu_name": torch.cuda.get_device_name(idx),
                "gpu_total_memory_mb": int(props.total_memory / (1024 ** 2)),
                "cuda_device_index": int(idx),
            }
        )
    return info


def _autocast_context(device: str, amp_enabled: bool):
    if device == "cuda" and amp_enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", enabled=False)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    amp_enabled: bool = False,
    scaler: Optional[torch.amp.GradScaler] = None,
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
        with _autocast_context(device=device, amp_enabled=amp_enabled):
            logits = model(xb)
            loss = criterion(logits, yb)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_max_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            optimizer.step()

        batch_size = int(yb.size(0))
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())
        total_seen += batch_size

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
    amp_enabled: bool = False,
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

        with _autocast_context(device=device, amp_enabled=amp_enabled):
            logits = model(xb)
            loss = criterion(logits, yb)

        preds = logits.argmax(dim=1)
        batch_size = int(yb.size(0))
        total_loss += float(loss.item()) * batch_size
        total_correct += int((preds == yb).sum().item())
        total_seen += batch_size

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


def run_training_stage(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    device: str,
    num_classes: int,
    epochs: int,
    stage_name: str,
    history: Optional[TrainingHistory] = None,
    best_state: Optional[Dict[str, Any]] = None,
    start_epoch: int = 0,
    amp_enabled: bool = False,
    scaler: Optional[torch.amp.GradScaler] = None,
    grad_clip_max_norm: Optional[float] = 1.0,
) -> Tuple[TrainingHistory, Dict[str, Any]]:
    if history is None:
        history = TrainingHistory()
    if best_state is None:
        best_state = {}

    best_val_macro_f1 = float(best_state.get("best_val_macro_f1", -math.inf))

    for epoch_idx in range(1, epochs + 1):
        global_epoch = start_epoch + epoch_idx
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
                epoch=int(global_epoch),
                stage=stage_name,
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
                "epoch": int(global_epoch),
                "stage": stage_name,
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

        print(
            f"[{stage_name}] "
            f"[Epoch {global_epoch:02d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f} "
            f"lr_backbone={backbone_lr:.6f} lr_head={head_lr:.6f}"
        )

    return history, best_state


def restore_best_weights(model: nn.Module, best_state: Dict[str, Any]) -> nn.Module:
    if not best_state or "model_state_dict" not in best_state:
        raise ValueError("best_state is empty or missing model_state_dict.")
    model.load_state_dict(best_state["model_state_dict"])
    return model


def load_checkpoint_for_resume(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint


def prepare_resume_state(
    checkpoint_path: Path,
    model: nn.Module,
    experiment_signature: str,
    map_location: str = "cpu",
) -> ResumeState:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if checkpoint.get("experiment_signature") != experiment_signature:
        raise ValueError("Resume checkpoint signature does not match current experiment signature.")

    history = training_history_from_dict(checkpoint.get("history"))
    best_state = {
        key: checkpoint[key]
        for key in [
            "epoch",
            "stage",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "scaler_state_dict",
            "best_val_macro_f1",
            "best_val_loss",
            "best_val_accuracy",
        ]
        if key in checkpoint
    }
    if "model_state_dict" in best_state:
        restore_best_weights(model, best_state)

    return ResumeState(
        history=history,
        best_state=best_state,
        completed_stage=checkpoint.get("stage"),
    )


@torch.no_grad()
def benchmark_inference(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    warmup_batches: int = 5,
    timed_batches: int = 20,
    amp_enabled: bool = False,
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
        with _autocast_context(device=device, amp_enabled=amp_enabled):
            _ = model(xb)
        if device == "cuda":
            torch.cuda.synchronize()

    timed = batches[:timed_batches]
    total_images = 0
    t0 = time.perf_counter()
    for xb in timed:
        xb = xb.to(device, non_blocking=True)
        with _autocast_context(device=device, amp_enabled=amp_enabled):
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
    try:
        import onnx  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("ONNX export requires the 'onnx' package to be installed.") from exc

    model.eval()
    dummy_input = torch.randn(*input_shape, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        str(export_path),
        export_params=True,
        opset_version=opset_version,
        dynamo=False,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )


def attempt_onnx_export(
    model: nn.Module,
    export_path: Path,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = "cpu",
    opset_version: int = 17,
) -> Dict[str, Any]:
    status = {"attempted": True, "succeeded": False, "path": str(export_path), "error": None}
    try:
        export_model_to_onnx(
            model=model,
            export_path=export_path,
            input_shape=input_shape,
            device=device,
            opset_version=opset_version,
        )
        status["succeeded"] = True
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
    return status


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

    return {"loss_curve": loss_curve_path, "accuracy_curve": acc_curve_path}




def training_history_from_dict(data: Optional[Dict[str, Any]]) -> TrainingHistory:
    history = TrainingHistory()
    if not data:
        return history
    for row in data.get("epochs", []):
        history.epochs.append(EpochMetrics(**row))
    return history

def build_metrics_payload(
    history: TrainingHistory,
    best_state: Dict[str, Any],
    test_metrics: Dict[str, Any],
    benchmark_metrics: Dict[str, float],
    parameter_count: int,
    trainable_parameter_count: int,
    model_size_mb: float,
    device_info: Dict[str, Any],
    onnx_export_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "history": history.to_dict(),
        "best_epoch": int(best_state.get("epoch", -1)),
        "best_stage": best_state.get("stage", None),
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
        "num_timed_batches": float(benchmark_metrics["num_timed_batches"]),
        "parameter_count": int(parameter_count),
        "trainable_parameter_count": int(trainable_parameter_count),
        "model_size_mb": float(model_size_mb),
        "device_info": device_info,
    }
    if onnx_export_status is not None:
        payload["onnx_export"] = onnx_export_status
    return payload


def save_report_metrics_copy(report_dir: Path, model_name: str, run_dir_name: str, metrics_payload: Dict[str, Any]) -> Path:
    ensure_dir(report_dir)
    report_path = report_dir / f"{model_name}_{run_dir_name}_metrics.json"
    atomic_save_json(report_path, metrics_payload)
    return report_path
