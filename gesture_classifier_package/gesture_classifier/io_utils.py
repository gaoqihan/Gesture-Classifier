from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch

from gesture_classifier.logger_utils import load_json
from gesture_classifier.models import (
    build_model_from_config_with_num_classes,
    expand_final_linear_layer,
)


def resolve_run_file(run_dir: str, filename: str) -> str:
    path = os.path.join(run_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def find_checkpoint_path(
    run_dir: str,
    preferred_name: str = "best_model.pt",
) -> str:
    preferred_path = os.path.join(run_dir, preferred_name)
    if os.path.isfile(preferred_path):
        return preferred_path

    candidates = []
    for name in os.listdir(run_dir):
        if name.endswith(".pt") or name.endswith(".pth"):
            candidates.append(os.path.join(run_dir, name))

    if len(candidates) == 0:
        raise FileNotFoundError(f"No checkpoint file found in run dir: {run_dir}")

    candidates.sort()
    return candidates[0]


def load_run_config_payload(run_dir: str) -> Dict[str, Any]:
    config_path = resolve_run_file(run_dir, "config.json")
    payload = load_json(config_path)

    if "config" not in payload:
        raise KeyError(f"'config' field missing from {config_path}")

    return payload


def load_run_config(run_dir: str) -> Dict[str, Any]:
    payload = load_run_config_payload(run_dir)
    return payload["config"]


def load_run_history(run_dir: str) -> Dict[str, Any]:
    history_path = resolve_run_file(run_dir, "history.json")
    return load_json(history_path)


def load_run_metrics(run_dir: str) -> Dict[str, Any]:
    metrics_path = resolve_run_file(run_dir, "metrics.json")
    return load_json(metrics_path)


def load_checkpoint(
    checkpoint_path: str,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint at {checkpoint_path} is not a dict.")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"'model_state_dict' missing in checkpoint: {checkpoint_path}")
    return checkpoint


def load_model_from_run(
    run_dir: str,
    num_classes: Optional[int] = None,
    device: str | torch.device = "cpu",
    checkpoint_name: str = "best_model.pt",
) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
        model, config, checkpoint
    """
    config = load_run_config(run_dir)

    if num_classes is None:
        labels = config.get("LABELS", None)
        if labels is None:
            raise ValueError(
                "num_classes not provided and LABELS missing from saved config."
            )
        num_classes = len(labels)

    model = build_model_from_config_with_num_classes(
        cfg=config,
        num_classes=num_classes,
    )

    checkpoint_path = find_checkpoint_path(run_dir, preferred_name=checkpoint_name)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    return model, config, checkpoint


def load_previous_run_artifacts(
    pretrained_run_dir: str,
    device: str | torch.device = "cpu",
    checkpoint_name: str = "best_model.pt",
) -> Dict[str, Any]:
    """
    For fine-tuning an existing model with the same class set.

    Returns:
        {
            "model": ...,
            "config": ...,
            "checkpoint": ...,
            "checkpoint_path": ...,
            "labels": ...,
            "num_classes": ...
        }
    """
    config = load_run_config(pretrained_run_dir)
    labels = config.get("LABELS", None)
    if labels is None:
        raise ValueError("Saved config does not contain LABELS.")

    num_classes = len(labels)

    model = build_model_from_config_with_num_classes(
        cfg=config,
        num_classes=num_classes,
    )

    checkpoint_path = find_checkpoint_path(
        pretrained_run_dir,
        preferred_name=checkpoint_name,
    )
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    return {
        "model": model,
        "config": config,
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "labels": labels,
        "num_classes": num_classes,
    }


def load_previous_run_for_expansion(
    previous_run_dir: str,
    new_num_classes: int,
    device: str | torch.device = "cpu",
    checkpoint_name: str = "best_model.pt",
) -> Dict[str, Any]:
    """
    For add-function / add-class workflow.
    Loads previous model, then expands final classifier layer.

    Returns:
        {
            "model": expanded_model,
            "config": ...,
            "checkpoint": ...,
            "checkpoint_path": ...,
            "old_labels": ...,
            "old_num_classes": ...,
            "new_num_classes": ...
        }
    """
    config = load_run_config(previous_run_dir)
    old_labels = config.get("LABELS", None)
    if old_labels is None:
        raise ValueError("Saved config does not contain LABELS.")

    old_num_classes = len(old_labels)
    if new_num_classes <= old_num_classes:
        raise ValueError(
            f"new_num_classes must be greater than old_num_classes "
            f"({new_num_classes} <= {old_num_classes})"
        )

    model = build_model_from_config_with_num_classes(
        cfg=config,
        num_classes=old_num_classes,
    )

    checkpoint_path = find_checkpoint_path(
        previous_run_dir,
        preferred_name=checkpoint_name,
    )
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model = expand_final_linear_layer(
        model=model,
        old_num_classes=old_num_classes,
        new_num_classes=new_num_classes,
    )
    model.to(device)

    return {
        "model": model,
        "config": config,
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "old_labels": old_labels,
        "old_num_classes": old_num_classes,
        "new_num_classes": new_num_classes,
    }


def merge_config_with_overrides(
    base_config: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    merged = dict(base_config)
    if overrides is not None:
        merged.update(overrides)
    return merged


def infer_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")