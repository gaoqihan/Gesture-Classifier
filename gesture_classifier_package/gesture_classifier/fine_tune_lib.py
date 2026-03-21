from __future__ import annotations

import copy
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from gesture_classifier.dataset import (
    GestureEpisodeDataset,
    gesture_collate_fn_stacked,
    make_train_val_test_subsets,
    summarize_subset_generic,
)
from gesture_classifier.io_utils import infer_device, load_previous_run_artifacts
from gesture_classifier.logger_utils import (
    build_run_summary_row,
    create_run_dir,
    save_full_run_artifacts,
    save_history_plot,
)
from gesture_classifier.models import (
    count_total_parameters,
    count_trainable_parameters,
    freeze_encoder_except_classifier,
    get_model_structure_string,
)
from gesture_classifier.train_utils import (
    EarlyStopping,
    build_training_components,
    print_metrics_summary,
    test_model,
    train_model,
)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_finetune_dataset_from_saved_config(
    data_root: str,
    labels,
    base_cfg: Dict[str, Any],
) -> GestureEpisodeDataset:
    body_landmark_used = base_cfg.get("BODY_LANDMARK_USED", None)

    dataset = GestureEpisodeDataset(
        data_root=data_root,
        labels=labels,
        video_input_enabled=bool(base_cfg.get("VIDEO_INPUT_ENABLED", True)),
        body_pose_input_enabled=bool(base_cfg.get("BODY_POSE_INPUT_ENABLED", True)),
        hand_pose_input_enabled=bool(base_cfg.get("HAND_POSE_INPUT_ENABLED", True)),
        body_landmark_used=body_landmark_used,
        video_size=tuple(base_cfg.get("VIDEO_SIZE", (224, 224))),
        video_normalize=bool(base_cfg.get("VIDEO_NORMALIZE", True)),
        video_pad_length=base_cfg.get("VIDEO_PAD_LENGTH", None),
        body_pad_length=base_cfg.get("BODY_PAD_LENGTH", None),
        hand_pad_length=base_cfg.get("HAND_PAD_LENGTH", None),
        truncate_if_longer=bool(base_cfg.get("TRUNCATE_IF_LONGER", True)),
        require_all_three_components=bool(
            base_cfg.get("REQUIRE_ALL_THREE_COMPONENTS", True)
        ),
        verbose_screening=True,
    )
    return dataset


def _limit_train_subset_to_n_samples(
    train_subset,
    n: int,
    random_seed: int,
):
    if n is None:
        return train_subset

    if n <= 0:
        raise ValueError("finetune_num_samples must be > 0")

    original_indices = list(train_subset.indices)
    if n >= len(original_indices):
        return train_subset

    rng = np.random.RandomState(random_seed)
    selected = rng.choice(original_indices, size=n, replace=False).tolist()

    new_subset = copy.copy(train_subset)
    new_subset.indices = selected
    return new_subset


def _build_finetune_config(
    base_cfg: Dict[str, Any],
    finetuning_data_root: str,
    pretrained_run_dir: str,
    finetune_num_samples: int,
    inference_only: bool,
    logger_root: str,
    finetune_run_note: str,
    random_seed: int,
    stratify: bool,
    finetune_epochs: int,
    finetune_learning_rate: float,
    finetune_weight_decay: float,
    finetune_batch_size: int,
    finetune_num_workers: int,
    finetune_use_class_weights: bool,
    finetune_use_lr_scheduler: bool,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    freeze_backbone: bool,
) -> Dict[str, Any]:
    cfg = dict(base_cfg)

    cfg.update(
        {
            "FINETUNING_DATA_ROOT": finetuning_data_root,
            "PRETRAINED_RUN_DIR": pretrained_run_dir,
            "FINETUNE_NUM_SAMPLES": finetune_num_samples,
            "INFERENCE_ONLY": inference_only,
            "LOGGER_ROOT": logger_root,
            "FINETUNE_RUN_NOTE": finetune_run_note,
            "RANDOM_SEED": random_seed,
            "STRATIFY_SPLIT": stratify,
            "FINETUNE_EPOCHS": finetune_epochs,
            "FINETUNE_LEARNING_RATE": finetune_learning_rate,
            "FINETUNE_WEIGHT_DECAY": finetune_weight_decay,
            "FINETUNE_BATCH_SIZE": finetune_batch_size,
            "FINETUNE_NUM_WORKERS": finetune_num_workers,
            "FINETUNE_USE_CLASS_WEIGHTS": finetune_use_class_weights,
            "FINETUNE_USE_LR_SCHEDULER": finetune_use_lr_scheduler,
            "EARLY_STOPPING_PATIENCE": early_stopping_patience,
            "EARLY_STOPPING_MIN_DELTA": early_stopping_min_delta,
            "FREEZE_BACKBONE": freeze_backbone,
            "RUN_KIND": "fine_tune",
        }
    )
    return cfg


def finetune_function(
    finetuning_data_root: str,
    pretrained_run_dir: str,
    finetune_num_samples: int,
    inference_only: bool = False,
    logger_root: str = "./logger_finetuning",
    finetune_run_note: str = "classifier_head_only",
    random_seed: int = 42,
    stratify: bool = True,
    finetune_epochs: int = 15,
    finetune_learning_rate: float = 1e-4,
    finetune_weight_decay: float = 1e-4,
    finetune_batch_size: int = 4,
    finetune_num_workers: int = 0,
    finetune_use_class_weights: bool = False,
    finetune_use_lr_scheduler: bool = False,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 1e-4,
    freeze_backbone: bool = True,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    set_global_seed(random_seed)
    torch_device = infer_device(device)

    loaded = load_previous_run_artifacts(
        pretrained_run_dir=pretrained_run_dir,
        device=torch_device,
    )

    model = loaded["model"]
    base_cfg = loaded["config"]
    labels = list(loaded["labels"])
    num_classes = int(loaded["num_classes"])

    if freeze_backbone and not inference_only:
        freeze_encoder_except_classifier(model)

    model_structure = get_model_structure_string(model)

    cfg = _build_finetune_config(
        base_cfg=base_cfg,
        finetuning_data_root=finetuning_data_root,
        pretrained_run_dir=pretrained_run_dir,
        finetune_num_samples=finetune_num_samples,
        inference_only=inference_only,
        logger_root=logger_root,
        finetune_run_note=finetune_run_note,
        random_seed=random_seed,
        stratify=stratify,
        finetune_epochs=finetune_epochs,
        finetune_learning_rate=finetune_learning_rate,
        finetune_weight_decay=finetune_weight_decay,
        finetune_batch_size=finetune_batch_size,
        finetune_num_workers=finetune_num_workers,
        finetune_use_class_weights=finetune_use_class_weights,
        finetune_use_lr_scheduler=finetune_use_lr_scheduler,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        freeze_backbone=freeze_backbone,
    )

    run_dir = create_run_dir(
        logger_root=logger_root,
        model_type=str(base_cfg.get("MODEL_TYPE", "CNN")),
        learning_rate=None if inference_only else finetune_learning_rate,
        epochs=None if inference_only else finetune_epochs,
        run_note=finetune_run_note,
        video_input_enabled=bool(base_cfg.get("VIDEO_INPUT_ENABLED", True)),
        body_pose_input_enabled=bool(base_cfg.get("BODY_POSE_INPUT_ENABLED", True)),
        hand_pose_input_enabled=bool(base_cfg.get("HAND_POSE_INPUT_ENABLED", True)),
    )

    dataset = _build_finetune_dataset_from_saved_config(
        data_root=finetuning_data_root,
        labels=labels,
        base_cfg=base_cfg,
    )

    train_subset, val_subset, test_subset, train_indices, val_indices, test_indices = (
        make_train_val_test_subsets(
            dataset=dataset,
            train_ratio=float(base_cfg.get("TRAIN_RATIO", 0.7)),
            val_ratio=float(base_cfg.get("VAL_RATIO", 0.15)),
            test_ratio=float(base_cfg.get("TEST_RATIO", 0.15)),
            random_seed=random_seed,
            stratify_split=stratify,
        )
    )

    train_subset = _limit_train_subset_to_n_samples(
        train_subset=train_subset,
        n=finetune_num_samples,
        random_seed=random_seed,
    )

    summarize_subset_generic(train_subset, dataset, labels, name="finetune_train")
    summarize_subset_generic(val_subset, dataset, labels, name="finetune_val")
    summarize_subset_generic(test_subset, dataset, labels, name="finetune_test")

    train_loader = DataLoader(
        train_subset,
        batch_size=finetune_batch_size,
        shuffle=not inference_only,
        num_workers=finetune_num_workers,
        collate_fn=gesture_collate_fn_stacked,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=finetune_batch_size,
        shuffle=False,
        num_workers=finetune_num_workers,
        collate_fn=gesture_collate_fn_stacked,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=finetune_batch_size,
        shuffle=False,
        num_workers=finetune_num_workers,
        collate_fn=gesture_collate_fn_stacked,
    )

    criterion, optimizer, scheduler, class_weights = build_training_components(
        model=model,
        device=torch_device,
        train_subset=train_subset,
        num_classes=num_classes,
        learning_rate=finetune_learning_rate,
        weight_decay=finetune_weight_decay,
        use_class_weights=finetune_use_class_weights,
        use_lr_scheduler=finetune_use_lr_scheduler,
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time_sec": [],
    }
    best_epoch = -1

    if not inference_only:
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            mode="min",
        )

        train_result = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=torch_device,
            epochs=finetune_epochs,
            labels=labels,
            video_input_enabled=bool(base_cfg.get("VIDEO_INPUT_ENABLED", True)),
            body_pose_input_enabled=bool(base_cfg.get("BODY_POSE_INPUT_ENABLED", True)),
            hand_pose_input_enabled=bool(base_cfg.get("HAND_POSE_INPUT_ENABLED", True)),
            scheduler=scheduler,
            early_stopping=early_stopping,
            selection_metric="val_loss",
        )

        model = train_result["model"]
        history = train_result["history"]
        best_epoch = train_result["best_epoch"]
        val_metrics = train_result["val_metrics"]
    else:
        val_metrics = test_model(
            model=model,
            test_loader=val_loader,
            criterion=criterion,
            device=torch_device,
            labels=labels,
            video_input_enabled=bool(base_cfg.get("VIDEO_INPUT_ENABLED", True)),
            body_pose_input_enabled=bool(base_cfg.get("BODY_POSE_INPUT_ENABLED", True)),
            hand_pose_input_enabled=bool(base_cfg.get("HAND_POSE_INPUT_ENABLED", True)),
        )

    test_metrics = test_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=torch_device,
        labels=labels,
        video_input_enabled=bool(base_cfg.get("VIDEO_INPUT_ENABLED", True)),
        body_pose_input_enabled=bool(base_cfg.get("BODY_POSE_INPUT_ENABLED", True)),
        hand_pose_input_enabled=bool(base_cfg.get("HAND_POSE_INPUT_ENABLED", True)),
    )

    print_metrics_summary(val_metrics, split_name="val")
    print_metrics_summary(test_metrics, split_name="test")

    plot_path = save_history_plot(
        run_dir=run_dir,
        history=history,
        model_type=str(base_cfg.get("MODEL_TYPE", "CNN")),
        learning_rate=None if inference_only else finetune_learning_rate,
        epochs=None if inference_only else finetune_epochs,
        video_input_enabled=bool(base_cfg.get("VIDEO_INPUT_ENABLED", True)),
        body_pose_input_enabled=bool(base_cfg.get("BODY_POSE_INPUT_ENABLED", True)),
        hand_pose_input_enabled=bool(base_cfg.get("HAND_POSE_INPUT_ENABLED", True)),
        run_note=finetune_run_note,
    )

    run_summary_row = build_run_summary_row(
        run_dir=run_dir,
        model_type=str(base_cfg.get("MODEL_TYPE", "CNN")),
        labels=labels,
        config=cfg,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        note=finetune_run_note,
    )

    artifact_paths = save_full_run_artifacts(
        logger_root=logger_root,
        run_dir=run_dir,
        config=cfg,
        model=model,
        model_structure=model_structure,
        history=history,
        labels=labels,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        checkpoint_name="best_model.pt",
        extra_checkpoint_data={
            "labels": labels,
            "num_classes": num_classes,
            "best_epoch": best_epoch,
            "train_indices": list(getattr(train_subset, "indices", train_indices)),
            "val_indices": list(val_indices),
            "test_indices": list(test_indices),
            "class_weights": None if class_weights is None else class_weights.detach().cpu(),
            "source_pretrained_run_dir": pretrained_run_dir,
            "run_kind": "fine_tune",
        },
        run_summary_row=run_summary_row,
    )

    artifact_paths["training_curve"] = plot_path

    return {
        "run_dir": run_dir,
        "artifact_paths": artifact_paths,
        "config": cfg,
        "labels": labels,
        "num_classes": num_classes,
        "model": model,
        "model_structure": model_structure,
        "history": history,
        "best_epoch": best_epoch,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "device": str(torch_device),
        "trainable_params": count_trainable_parameters(model),
        "total_params": count_total_parameters(model),
    }