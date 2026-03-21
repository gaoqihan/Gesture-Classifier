from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_for_filename(text: str) -> str:
    safe = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        elif ch in {" ", "."}:
            safe.append("_")
    out = "".join(safe).strip("_")
    return out or "run"


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_modalities_tag(
    video_input_enabled: bool,
    body_pose_input_enabled: bool,
    hand_pose_input_enabled: bool,
) -> str:
    parts: List[str] = []
    if video_input_enabled:
        parts.append("video")
    if body_pose_input_enabled:
        parts.append("body")
    if hand_pose_input_enabled:
        parts.append("hand")
    return "+".join(parts) if parts else "none"


def build_run_name(
    model_type: str,
    learning_rate: Optional[float] = None,
    epochs: Optional[int] = None,
    run_note: str = "",
    video_input_enabled: bool = True,
    body_pose_input_enabled: bool = True,
    hand_pose_input_enabled: bool = True,
) -> str:
    model_tag = sanitize_for_filename(str(model_type).upper())
    modality_tag = build_modalities_tag(
        video_input_enabled=video_input_enabled,
        body_pose_input_enabled=body_pose_input_enabled,
        hand_pose_input_enabled=hand_pose_input_enabled,
    )
    note_tag = sanitize_for_filename(run_note) if run_note else "run"

    parts = [timestamp_now(), model_tag, modality_tag]

    if learning_rate is not None:
        parts.append(f"lr_{learning_rate:g}")

    if epochs is not None:
        parts.append(f"ep_{epochs}")

    parts.append(note_tag)
    return "__".join(parts)


def create_run_dir(
    logger_root: str,
    model_type: str,
    learning_rate: Optional[float] = None,
    epochs: Optional[int] = None,
    run_note: str = "",
    video_input_enabled: bool = True,
    body_pose_input_enabled: bool = True,
    hand_pose_input_enabled: bool = True,
) -> str:
    ensure_dir(logger_root)

    run_name = build_run_name(
        model_type=model_type,
        learning_rate=learning_rate,
        epochs=epochs,
        run_note=run_note,
        video_input_enabled=video_input_enabled,
        body_pose_input_enabled=body_pose_input_enabled,
        hand_pose_input_enabled=hand_pose_input_enabled,
    )

    run_dir = os.path.join(logger_root, run_name)
    ensure_dir(run_dir)
    return run_dir


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(data), f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(text: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_config(
    run_dir: str,
    config: Dict[str, Any],
    model_structure: Optional[str] = None,
    extra_info: Optional[Dict[str, Any]] = None,
) -> str:
    payload = {
        "config": _to_jsonable(config),
    }

    if model_structure is not None:
        payload["model_structure"] = model_structure

    if extra_info is not None:
        payload["extra_info"] = _to_jsonable(extra_info)

    path = os.path.join(run_dir, "config.json")
    save_json(payload, path)
    return path


def save_history(
    run_dir: str,
    history: Dict[str, Any],
) -> str:
    path = os.path.join(run_dir, "history.json")
    save_json(history, path)
    return path


def save_metrics(
    run_dir: str,
    val_metrics: Dict[str, Any],
    test_metrics: Optional[Dict[str, Any]] = None,
    train_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    payload = {
        "val_metrics": _to_jsonable(val_metrics),
        "test_metrics": _to_jsonable(test_metrics) if test_metrics is not None else None,
        "train_metrics": _to_jsonable(train_metrics) if train_metrics is not None else None,
    }
    path = os.path.join(run_dir, "metrics.json")
    save_json(payload, path)
    return path


def save_classification_reports(
    run_dir: str,
    val_metrics: Dict[str, Any],
    test_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    out = {}

    val_report = val_metrics.get("classification_report", None)
    if val_report is not None:
        val_path = os.path.join(run_dir, "classification_report_val.txt")
        save_text(str(val_report), val_path)
        out["val"] = val_path

    if test_metrics is not None:
        test_report = test_metrics.get("classification_report", None)
        if test_report is not None:
            test_path = os.path.join(run_dir, "classification_report_test.txt")
            save_text(str(test_report), test_path)
            out["test"] = test_path

    return out


def save_confusion_matrix_csv(
    cm: np.ndarray,
    labels: List[str],
    path: str,
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + list(labels))
        for i, row in enumerate(cm):
            writer.writerow([labels[i]] + list(map(int, row)))


def save_confusion_matrices(
    run_dir: str,
    labels: List[str],
    val_metrics: Dict[str, Any],
    test_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    out = {}

    val_cm = val_metrics.get("confusion_matrix", None)
    if val_cm is not None:
        val_path = os.path.join(run_dir, "confusion_matrix_val.csv")
        save_confusion_matrix_csv(np.asarray(val_cm), labels, val_path)
        out["val"] = val_path

    if test_metrics is not None:
        test_cm = test_metrics.get("confusion_matrix", None)
        if test_cm is not None:
            test_path = os.path.join(run_dir, "confusion_matrix_test.csv")
            save_confusion_matrix_csv(np.asarray(test_cm), labels, test_path)
            out["test"] = test_path

    return out


def save_model_checkpoint(
    run_dir: str,
    model: torch.nn.Module,
    checkpoint_name: str = "best_model.pt",
    extra_checkpoint_data: Optional[Dict[str, Any]] = None,
) -> str:
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }

    if extra_checkpoint_data is not None:
        checkpoint.update(extra_checkpoint_data)

    path = os.path.join(run_dir, checkpoint_name)
    torch.save(checkpoint, path)
    return path


def plot_training_history(
    history: Dict[str, Any],
    path: str,
    title: str,
) -> None:
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history.get("train_loss", []), label="train_loss")
    plt.plot(epochs, history.get("val_loss", []), label="val_loss")
    plt.plot(epochs, history.get("train_acc", []), label="train_acc")
    plt.plot(epochs, history.get("val_acc", []), label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_history_plot(
    run_dir: str,
    history: Dict[str, Any],
    model_type: str,
    learning_rate: Optional[float],
    epochs: Optional[int],
    video_input_enabled: bool,
    body_pose_input_enabled: bool,
    hand_pose_input_enabled: bool,
    run_note: str = "",
) -> str:
    modality_tag = build_modalities_tag(
        video_input_enabled=video_input_enabled,
        body_pose_input_enabled=body_pose_input_enabled,
        hand_pose_input_enabled=hand_pose_input_enabled,
    )

    title_parts = [str(model_type).upper(), modality_tag]
    if learning_rate is not None:
        title_parts.append(f"lr={learning_rate:g}")
    if epochs is not None:
        title_parts.append(f"epochs={epochs}")
    if run_note:
        title_parts.append(run_note)

    title = " | ".join(title_parts)
    path = os.path.join(run_dir, "training_curve.png")
    plot_training_history(history=history, path=path, title=title)
    return path


def append_to_meta_csv(
    logger_root: str,
    row: Dict[str, Any],
    filename: str = "meta_runs.csv",
) -> str:
    ensure_dir(logger_root)
    csv_path = os.path.join(logger_root, filename)

    row = {str(k): _to_jsonable(v) for k, v in row.items()}

    existing_rows: List[Dict[str, Any]] = []
    fieldnames = set(row.keys())

    if os.path.isfile(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_rows.append(dict(r))
                fieldnames.update(r.keys())

    fieldnames = list(sorted(fieldnames))

    existing_rows.append({k: row.get(k, "") for k in fieldnames})

    normalized_rows = []
    for r in existing_rows:
        normalized_rows.append({k: r.get(k, "") for k in fieldnames})

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)

    return csv_path


def save_full_run_artifacts(
    logger_root: str,
    run_dir: str,
    config: Dict[str, Any],
    model: torch.nn.Module,
    model_structure: str,
    history: Dict[str, Any],
    labels: List[str],
    val_metrics: Dict[str, Any],
    test_metrics: Optional[Dict[str, Any]] = None,
    checkpoint_name: str = "best_model.pt",
    extra_checkpoint_data: Optional[Dict[str, Any]] = None,
    run_summary_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    outputs: Dict[str, str] = {}

    outputs["config_json"] = save_config(
        run_dir=run_dir,
        config=config,
        model_structure=model_structure,
    )

    outputs["history_json"] = save_history(
        run_dir=run_dir,
        history=history,
    )

    outputs["metrics_json"] = save_metrics(
        run_dir=run_dir,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )

    report_paths = save_classification_reports(
        run_dir=run_dir,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
    for k, v in report_paths.items():
        outputs[f"classification_report_{k}"] = v

    cm_paths = save_confusion_matrices(
        run_dir=run_dir,
        labels=labels,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
    for k, v in cm_paths.items():
        outputs[f"confusion_matrix_{k}"] = v

    outputs["checkpoint"] = save_model_checkpoint(
        run_dir=run_dir,
        model=model,
        checkpoint_name=checkpoint_name,
        extra_checkpoint_data=extra_checkpoint_data,
    )

    if run_summary_row is not None:
        outputs["meta_csv"] = append_to_meta_csv(
            logger_root=logger_root,
            row=run_summary_row,
        )

    return outputs


def build_run_summary_row(
    run_dir: str,
    model_type: str,
    labels: List[str],
    config: Dict[str, Any],
    val_metrics: Dict[str, Any],
    test_metrics: Optional[Dict[str, Any]] = None,
    note: str = "",
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "run_dir": run_dir,
        "model_type": model_type,
        "labels": "|".join(labels),
        "note": note,
        "val_acc": val_metrics.get("acc", ""),
        "val_loss": val_metrics.get("loss", ""),
        "val_macro_f1": val_metrics.get("macro_f1", ""),
    }

    if test_metrics is not None:
        row["test_acc"] = test_metrics.get("acc", "")
        row["test_loss"] = test_metrics.get("loss", "")
        row["test_macro_f1"] = test_metrics.get("macro_f1", "")

    for k, v in config.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            row[f"cfg_{k}"] = v
        else:
            row[f"cfg_{k}"] = json.dumps(_to_jsonable(v), ensure_ascii=False)

    return row