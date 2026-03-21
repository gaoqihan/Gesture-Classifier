from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


def move_inputs_to_device(
    batch_inputs,
    device: torch.device,
    video_input_enabled: bool,
    body_pose_input_enabled: bool,
    hand_pose_input_enabled: bool,
) -> Dict[str, Optional[torch.Tensor]]:
    """
    Supports the dataset/collate structure from dataset.py.

    Expected order in batch_inputs:
    - video tensor if enabled
    - body tensor if enabled
    - hand tensor if enabled
    - body_input_shape list if body enabled
    """
    idx = 0

    video = None
    body = None
    hand = None

    if video_input_enabled:
        video = batch_inputs[idx].to(device)
        idx += 1

    if body_pose_input_enabled:
        body = batch_inputs[idx].to(device)
        idx += 1

    if hand_pose_input_enabled:
        hand = batch_inputs[idx].to(device)
        idx += 1

    # Optional trailing body_input_shape, ignored for model forward
    if body_pose_input_enabled and idx < len(batch_inputs):
        _ = batch_inputs[idx]

    return {
        "video": video,
        "body": body,
        "hand": hand,
    }


def forward_model_from_batch_inputs(
    model: nn.Module,
    batch_inputs,
    device: torch.device,
    video_input_enabled: bool,
    body_pose_input_enabled: bool,
    hand_pose_input_enabled: bool,
) -> torch.Tensor:
    model_inputs = move_inputs_to_device(
        batch_inputs=batch_inputs,
        device=device,
        video_input_enabled=video_input_enabled,
        body_pose_input_enabled=body_pose_input_enabled,
        hand_pose_input_enabled=hand_pose_input_enabled,
    )

    logits = model(
        video=model_inputs["video"],
        body=model_inputs["body"],
        hand=model_inputs["hand"],
    )
    return logits


def compute_class_weights_from_subset(
    subset,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)

    for idx in subset.indices:
        label_idx = subset.dataset.samples[idx]["label_idx"]
        counts[label_idx] += 1.0

    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return weights


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Optimizer:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found when creating optimizer.")

    return Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)


def create_scheduler(
    optimizer: Optimizer,
    factor: float = 0.5,
    patience: int = 3,
    min_lr: float = 1e-6,
):
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
        min_lr=min_lr,
    )


def build_criterion(
    class_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    return nn.CrossEntropyLoss(weight=class_weights)


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[Optimizer],
    criterion: nn.Module,
    device: torch.device,
    train: bool,
    video_input_enabled: bool,
    body_pose_input_enabled: bool,
    hand_pose_input_enabled: bool,
) -> Dict[str, Any]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels: List[int] = []
    all_preds: List[int] = []

    for batch_inputs, labels, _session_ids in loader:
        labels = labels.to(device)

        with torch.set_grad_enabled(train):
            logits = forward_model_from_batch_inputs(
                model=model,
                batch_inputs=batch_inputs,
                device=device,
                video_input_enabled=video_input_enabled,
                body_pose_input_enabled=body_pose_input_enabled,
                hand_pose_input_enabled=hand_pose_input_enabled,
            )

            loss = criterion(logits, labels)

            if train:
                if optimizer is None:
                    raise ValueError("optimizer cannot be None when train=True")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = torch.argmax(logits, dim=1)
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(total_samples, 1)
    acc = accuracy_score(all_labels, all_preds) if total_samples > 0 else 0.0

    return {
        "loss": float(avg_loss),
        "acc": float(acc),
        "labels": all_labels,
        "preds": all_preds,
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    labels: Sequence[str],
    video_input_enabled: bool,
    body_pose_input_enabled: bool,
    hand_pose_input_enabled: bool,
) -> Dict[str, Any]:
    metrics = run_one_epoch(
        model=model,
        loader=loader,
        optimizer=None,
        criterion=criterion,
        device=device,
        train=False,
        video_input_enabled=video_input_enabled,
        body_pose_input_enabled=body_pose_input_enabled,
        hand_pose_input_enabled=hand_pose_input_enabled,
    )

    y_true = metrics["labels"]
    y_pred = metrics["preds"]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        average=None,
        zero_division=0,
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
    )

    clf_report = classification_report(
        y_true,
        y_pred,
        target_names=list(labels),
        zero_division=0,
        digits=4,
    )

    per_class = {}
    for i, label_name in enumerate(labels):
        per_class[label_name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    metrics.update(
        {
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "confusion_matrix": cm,
            "classification_report": clf_report,
            "per_class": per_class,
        }
    )
    return metrics


class EarlyStopping:
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ):
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_value: Optional[float] = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def step(self, current_value: float) -> bool:
        if self.best_value is None:
            self.best_value = current_value
            self.num_bad_epochs = 0
            return False

        improved = False
        if self.mode == "min":
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current_value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.should_stop = True

        return self.should_stop


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epochs: int,
    labels: Sequence[str],
    video_input_enabled: bool,
    body_pose_input_enabled: bool,
    hand_pose_input_enabled: bool,
    scheduler=None,
    early_stopping: Optional[EarlyStopping] = None,
    selection_metric: str = "val_loss",
) -> Dict[str, Any]:
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time_sec": [],
    }

    best_epoch = -1
    best_state_dict = copy.deepcopy(model.state_dict())

    if selection_metric == "val_loss":
        best_metric_value = float("inf")
    elif selection_metric == "val_acc":
        best_metric_value = float("-inf")
    else:
        raise ValueError("selection_metric must be 'val_loss' or 'val_acc'")

    for epoch in range(epochs):
        start_time = time.time()

        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train=True,
            video_input_enabled=video_input_enabled,
            body_pose_input_enabled=body_pose_input_enabled,
            hand_pose_input_enabled=hand_pose_input_enabled,
        )

        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            criterion=criterion,
            device=device,
            train=False,
            video_input_enabled=video_input_enabled,
            body_pose_input_enabled=body_pose_input_enabled,
            hand_pose_input_enabled=hand_pose_input_enabled,
        )

        if scheduler is not None:
            scheduler.step(val_metrics["loss"])

        epoch_time = time.time() - start_time

        history["train_loss"].append(float(train_metrics["loss"]))
        history["train_acc"].append(float(train_metrics["acc"]))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_acc"].append(float(val_metrics["acc"]))
        history["epoch_time_sec"].append(float(epoch_time))

        improved = False
        if selection_metric == "val_loss":
            if val_metrics["loss"] < best_metric_value:
                best_metric_value = float(val_metrics["loss"])
                improved = True
        else:
            if val_metrics["acc"] > best_metric_value:
                best_metric_value = float(val_metrics["acc"])
                improved = True

        if improved:
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} | "
            f"time={epoch_time:.2f}s"
        )

        if early_stopping is not None:
            stop = early_stopping.step(val_metrics["loss"])
            if stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state_dict)

    final_val_metrics = evaluate_model(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        labels=labels,
        video_input_enabled=video_input_enabled,
        body_pose_input_enabled=body_pose_input_enabled,
        hand_pose_input_enabled=hand_pose_input_enabled,
    )

    return {
        "model": model,
        "history": history,
        "best_epoch": int(best_epoch),
        "best_selection_metric": float(best_metric_value),
        "val_metrics": final_val_metrics,
        "best_state_dict": best_state_dict,
    }


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    labels: Sequence[str],
    video_input_enabled: bool,
    body_pose_input_enabled: bool,
    hand_pose_input_enabled: bool,
) -> Dict[str, Any]:
    return evaluate_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        labels=labels,
        video_input_enabled=video_input_enabled,
        body_pose_input_enabled=body_pose_input_enabled,
        hand_pose_input_enabled=hand_pose_input_enabled,
    )


def print_metrics_summary(
    metrics: Dict[str, Any],
    split_name: str = "eval",
) -> None:
    print(f"\n=== {split_name} metrics ===")
    print(f"loss: {metrics['loss']:.4f}")
    print(f"acc: {metrics['acc']:.4f}")
    print(f"macro_precision: {metrics['macro_precision']:.4f}")
    print(f"macro_recall: {metrics['macro_recall']:.4f}")
    print(f"macro_f1: {metrics['macro_f1']:.4f}")


def build_training_components(
    model: nn.Module,
    device: torch.device,
    train_subset,
    num_classes: int,
    learning_rate: float,
    weight_decay: float,
    use_class_weights: bool,
    use_lr_scheduler: bool,
):
    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights_from_subset(
            subset=train_subset,
            num_classes=num_classes,
            device=device,
        )

    criterion = build_criterion(class_weights=class_weights)
    optimizer = create_optimizer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = None
    if use_lr_scheduler:
        scheduler = create_scheduler(optimizer)

    return criterion, optimizer, scheduler, class_weights