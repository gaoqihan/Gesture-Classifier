from __future__ import annotations

import random
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from gesture_classifier.dataset import (
    GestureEpisodeDataset,
    gesture_collate_fn_stacked,
    make_train_val_test_subsets_for_new_class,
    summarize_subset_generic,
)
from gesture_classifier.io_utils import infer_device, load_run_config, load_previous_run_for_expansion
from gesture_classifier.logger_utils import (
    build_run_summary_row,
    create_run_dir,
    save_full_run_artifacts,
    save_history_plot,
)
from gesture_classifier.models import (
    build_model_from_config_with_num_classes,
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


def _build_dataset_from_config(
    data_root: str,
    labels,
    cfg: Dict[str, Any],
) -> GestureEpisodeDataset:
    return GestureEpisodeDataset(
        data_root=data_root,
        labels=labels,
        video_input_enabled=bool(cfg.get("VIDEO_INPUT_ENABLED", True)),
        body_pose_input_enabled=bool(cfg.get("BODY_POSE_INPUT_ENABLED", True)),
        hand_pose_input_enabled=bool(cfg.get("HAND_POSE_INPUT_ENABLED", True)),
        body_landmark_used=cfg.get("BODY_LANDMARK_USED", None),
        video_size=tuple(cfg.get("VIDEO_SIZE", (224, 224))),
        video_normalize=bool(cfg.get("VIDEO_NORMALIZE", True)),
        video_pad_length=cfg.get("VIDEO_PAD_LENGTH", None),
        body_pad_length=cfg.get("BODY_PAD_LENGTH", None),
        hand_pad_length=cfg.get("HAND_PAD_LENGTH", None),
        truncate_if_longer=bool(cfg.get("TRUNCATE_IF_LONGER", True)),
        require_all_three_components=bool(
            cfg.get("REQUIRE_ALL_THREE_COMPONENTS", True)
        ),
        verbose_screening=True,
    )


def _build_add_function_config(
    base_cfg: Dict[str, Any],
    data_root: str,
    labels,
    new_class_name: str,
    new_class_train_samples: Optional[int],
    init_mode: str,
    previous_run_dir: Optional[str],
    logger_root: str,
    run_note: str,
    random_seed: int,
    stratify: bool,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    num_workers: int,
    use_class_weights: bool,
    use_lr_scheduler: bool,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    freeze_backbone: bool,
) -> Dict[str, Any]:
    cfg = dict(base_cfg)
    cfg.update(
        {
            "DATA_ROOT": data_root,
            "LABELS": list(labels),
            "NEW_CLASS_NAME": new_class_name,
            "NEW_CLASS_TRAIN_SAMPLES": new_class_train_samples,
            "INIT_MODE": init_mode,
            "PREVIOUS_RUN_DIR": previous_run_dir,
            "LOGGER_ROOT": logger_root,
            "RUN_NOTE": run_note,
            "RANDOM_SEED": random_seed,
            "STRATIFY_SPLIT": stratify,
            "EPOCHS": epochs,
            "LEARNING_RATE": learning_rate,
            "WEIGHT_DECAY": weight_decay,
            "BATCH_SIZE": batch_size,
            "NUM_WORKERS": num_workers,
            "USE_CLASS_WEIGHTS": use_class_weights,
            "USE_LR_SCHEDULER": use_lr_scheduler,
            "EARLY_STOPPING_PATIENCE": early_stopping_patience,
            "EARLY_STOPPING_MIN_DELTA": early_stopping_min_delta,
            "FREEZE_BACKBONE": freeze_backbone,
            "RUN_KIND": "add_function",
        }
    )
    return cfg


def add_function(
    data_root: str,
    new_class_name: str,
    new_class_train_samples: Optional[int] = None,
    init_mode: str = "expand_from_checkpoint",
    previous_run_dir: Optional[str] = None,
    logger_root: str = "./logger_add_function",
    run_note: str = "add_new_class",
    random_seed: int = 42,
    stratify: bool = True,
    epochs: int = 15,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 4,
    num_workers: int = 0,
    use_class_weights: bool = False,
    use_lr_scheduler: bool = False,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 1e-4,
    freeze_backbone: bool = True,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    if init_mode not in {"expand_from_checkpoint", "scratch"}:
        raise ValueError(
            "init_mode must be one of {'expand_from_checkpoint', 'scratch'}"
        )

    if init_mode == "expand_from_checkpoint" and not previous_run_dir:
        raise ValueError(
            "previous_run_dir is required when init_mode='expand_from_checkpoint'"
        )

    set_global_seed(random_seed)
    torch_device = infer_device(device)

    if init_mode == "expand_from_checkpoint":
        base_cfg = load_run_config(previous_run_dir)
        old_labels = list(base_cfg.get("LABELS", []))
        if len(old_labels) == 0:
            raise ValueError("Saved config has empty or missing LABELS.")

        if new_class_name in old_labels:
            raise ValueError(
                f"new_class_name '{new_class_name}' already exists in old labels."
            )

        labels = old_labels + [new_class_name]

        loaded = load_previous_run_for_expansion(
            previous_run_dir=previous_run_dir,
            new_num_classes=len(labels),
            device=torch_device,
        )
        model = loaded["model"]
        base_cfg = dict(loaded["config"])
        old_labels = list(loaded["old_labels"])
        old_num_classes = int(loaded["old_num_classes"])
    else:
        base_cfg = {
            "MODEL_TYPE": "CNN",
            "VIDEO_INPUT_ENABLED": True,
            "BODY_POSE_INPUT_ENABLED": True,
            "HAND_POSE_INPUT_ENABLED": True,
            "VIDEO_FEATURE_DIM": 64,
            "BODY_FEATURE_DIM": 32,
            "HAND_FEATURE_DIM": 32,
            "POSE_HIDDEN_DIM": 128,
            "POSE_DROPOUT": 0.1,
            "CLASSIFIER_DROPOUT": 0.2,
            "TEMPORAL_HIDDEN_DIM": 128,
            "LSTM_HIDDEN_DIM": 128,
            "LSTM_NUM_LAYERS": 1,
            "LSTM_BIDIRECTIONAL": False,
            "VIDEO_SIZE": (224, 224),
            "VIDEO_NORMALIZE": True,
            "VIDEO_PAD_LENGTH": None,
            "BODY_PAD_LENGTH": None,
            "HAND_PAD_LENGTH": None,
            "TRUNCATE_IF_LONGER": True,
            "REQUIRE_ALL_THREE_COMPONENTS": True,
            "TRAIN_RATIO": 0.7,
            "VAL_RATIO": 0.15,
            "TEST_RATIO": 0.15,
            "BODY_LANDMARK_USED": None,
        }
        old_labels = []
        old_num_classes = 0
        labels = [new_class_name]

        model = build_model_from_config_with_num_classes(
            cfg=base_cfg,
            num_classes=len(labels),
        ).to(torch_device)

    num_classes = len(labels)

    if freeze_backbone and init_mode == "expand_from_checkpoint":
        freeze_encoder_except_classifier(model)

    model_structure = get_model_structure_string(model)

    cfg = _build_add_function_config(
        base_cfg=base_cfg,
        data_root=data_root,
        labels=labels,
        new_class_name=new_class_name,
        new_class_train_samples=new_class_train_samples,
        init_mode=init_mode,
        previous_run_dir=previous_run_dir,
        logger_root=logger_root,
        run_note=run_note,
        random_seed=random_seed,
        stratify=stratify,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        num_workers=num_workers,
        use_class_weights=use_class_weights,
        use_lr_scheduler=use_lr_scheduler,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        freeze_backbone=freeze_backbone,
    )

    run_dir = create_run_dir(
        logger_root=logger_root,
        model_type=str(base_cfg.get("MODEL_TYPE", "CNN")),
        learning_rate=learning_rate,
        epochs=epochs,
        run_note=run_note,
        video_input_enabled=bool(base_cfg.get("VIDEO_INPUT_ENABLED", True)),
        body_pose_input_enabled=bool(base_cfg.get("BODY_POSE_INPUT_ENABLED", True)),
        hand_pose_input_enabled=bool(base_cfg.get("HAND_POSE_INPUT_ENABLED", True)),
    )

    dataset = _build_dataset_from_config(
        data_root=data_root,
        labels=labels,
        cfg=cfg,
    )

    train_subset, val_subset, test_subset, train_indices, val_indices, test_indices = (
        make_train_val_test_subsets_for_new_class(
            dataset=dataset,
            labels=labels,
            new_class_name=new_class_name,
            train_ratio=float(cfg.get("TRAIN_RATIO", 0.7)),
            val_ratio=float(cfg.get("VAL_RATIO", 0.15)),
            test_ratio=float(cfg.get("TEST_RATIO", 0.15)),
            random_seed=random_seed,
            stratify=stratify,
            new_class_train_samples=new_class_train_samples,
        )
    )

    summarize_subset_generic(train_subset, dataset, labels, name="add_function_train")
    summarize_subset_generic(val_subset, dataset, labels, name="add_function_val")
    summarize_subset_generic(test_subset, dataset, labels, name="add_function_test")

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=gesture_collate_fn_stacked,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=gesture_collate_fn_stacked,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=gesture_collate_fn_stacked,
    )

    criterion, optimizer, scheduler, class_weights = build_training_components(
        model=model,
        device=torch_device,
        train_subset=train_subset,
        num_classes=num_classes,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        use_class_weights=use_class_weights,
        use_lr_scheduler=use_lr_scheduler,
    )

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
        epochs=epochs,
        labels=labels,
        video_input_enabled=bool(cfg.get("VIDEO_INPUT_ENABLED", True)),
        body_pose_input_enabled=bool(cfg.get("BODY_POSE_INPUT_ENABLED", True)),
        hand_pose_input_enabled=bool(cfg.get("HAND_POSE_INPUT_ENABLED", True)),
        scheduler=scheduler,
        early_stopping=early_stopping,
        selection_metric="val_loss",
    )

    model = train_result["model"]
    history = train_result["history"]
    best_epoch = train_result["best_epoch"]
    val_metrics = train_result["val_metrics"]

    test_metrics = test_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=torch_device,
        labels=labels,
        video_input_enabled=bool(cfg.get("VIDEO_INPUT_ENABLED", True)),
        body_pose_input_enabled=bool(cfg.get("BODY_POSE_INPUT_ENABLED", True)),
        hand_pose_input_enabled=bool(cfg.get("HAND_POSE_INPUT_ENABLED", True)),
    )

    print_metrics_summary(val_metrics, split_name="val")
    print_metrics_summary(test_metrics, split_name="test")

    plot_path = save_history_plot(
        run_dir=run_dir,
        history=history,
        model_type=str(cfg.get("MODEL_TYPE", "CNN")),
        learning_rate=learning_rate,
        epochs=epochs,
        video_input_enabled=bool(cfg.get("VIDEO_INPUT_ENABLED", True)),
        body_pose_input_enabled=bool(cfg.get("BODY_POSE_INPUT_ENABLED", True)),
        hand_pose_input_enabled=bool(cfg.get("HAND_POSE_INPUT_ENABLED", True)),
        run_note=run_note,
    )

    run_summary_row = build_run_summary_row(
        run_dir=run_dir,
        model_type=str(cfg.get("MODEL_TYPE", "CNN")),
        labels=labels,
        config=cfg,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        note=run_note,
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
            "old_labels": old_labels,
            "old_num_classes": old_num_classes,
            "new_class_name": new_class_name,
            "best_epoch": best_epoch,
            "train_indices": list(train_indices),
            "val_indices": list(val_indices),
            "test_indices": list(test_indices),
            "class_weights": None if class_weights is None else class_weights.detach().cpu(),
            "source_previous_run_dir": previous_run_dir,
            "init_mode": init_mode,
            "run_kind": "add_function",
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