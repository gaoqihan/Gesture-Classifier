from __future__ import annotations

import argparse
import json

from gesture_classifier.fine_tune_lib import finetune_function


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune an existing gesture classifier on current data."
    )

    parser.add_argument(
        "--finetuning_data_root",
        required=True,
        type=str,
        help="Root folder containing fine-tuning data arranged by label folders.",
    )
    parser.add_argument(
        "--pretrained_run_dir",
        required=True,
        type=str,
        help="Run directory containing config.json and best_model.pt from a previous run.",
    )
    parser.add_argument(
        "--finetune_num_samples",
        required=True,
        type=int,
        help="Maximum number of training samples to use from the fine-tuning split.",
    )

    parser.add_argument(
        "--inference_only",
        default=False,
        type=str2bool,
        help="If true, skip training and only evaluate the loaded checkpoint on the new split.",
    )
    parser.add_argument(
        "--logger_root",
        default="./logger_finetuning",
        type=str,
        help="Root folder where fine-tuning logs and checkpoints will be saved.",
    )
    parser.add_argument(
        "--finetune_run_note",
        default="classifier_head_only",
        type=str,
        help="Free-text note used in run naming and metadata.",
    )
    parser.add_argument(
        "--random_seed",
        default=42,
        type=int,
        help="Random seed.",
    )
    parser.add_argument(
        "--stratify",
        default=True,
        type=str2bool,
        help="Whether to stratify train/val/test split by class.",
    )

    parser.add_argument(
        "--finetune_epochs",
        default=200,
        type=int,
        help="Fine-tuning epochs.",
    )
    parser.add_argument(
        "--finetune_learning_rate",
        default=1e-2,
        type=float,
        help="Learning rate for fine-tuning.",
    )
    parser.add_argument(
        "--finetune_weight_decay",
        default=1e-4,
        type=float,
        help="Weight decay for optimizer.",
    )
    parser.add_argument(
        "--finetune_batch_size",
        default=4,
        type=int,
        help="Batch size.",
    )
    parser.add_argument(
        "--finetune_num_workers",
        default=0,
        type=int,
        help="DataLoader num_workers.",
    )

    parser.add_argument(
        "--finetune_use_class_weights",
        default=False,
        type=str2bool,
        help="Whether to use inverse-frequency class weights.",
    )
    parser.add_argument(
        "--finetune_use_lr_scheduler",
        default=True,
        type=str2bool,
        help="Whether to use ReduceLROnPlateau scheduler.",
    )

    parser.add_argument(
        "--early_stopping_patience",
        default=20,
        type=int,
        help="Early stopping patience on validation loss.",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        default=1e-4,
        type=float,
        help="Minimum validation loss improvement for early stopping.",
    )
    parser.add_argument(
        "--freeze_backbone",
        default=True,
        type=str2bool,
        help="If true, only train classifier head.",
    )
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="Torch device string, e.g. cuda, cuda:0, cpu. Defaults to auto-detect.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    result = finetune_function(
        finetuning_data_root=args.finetuning_data_root,
        pretrained_run_dir=args.pretrained_run_dir,
        finetune_num_samples=args.finetune_num_samples,
        inference_only=args.inference_only,
        logger_root=args.logger_root,
        finetune_run_note=args.finetune_run_note,
        random_seed=args.random_seed,
        stratify=args.stratify,
        finetune_epochs=args.finetune_epochs,
        finetune_learning_rate=args.finetune_learning_rate,
        finetune_weight_decay=args.finetune_weight_decay,
        finetune_batch_size=args.finetune_batch_size,
        finetune_num_workers=args.finetune_num_workers,
        finetune_use_class_weights=args.finetune_use_class_weights,
        finetune_use_lr_scheduler=args.finetune_use_lr_scheduler,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        freeze_backbone=args.freeze_backbone,
        device=args.device,
    )

    summary = {
        "run_dir": result["run_dir"],
        "device": result["device"],
        "num_classes": result["num_classes"],
        "trainable_params": result["trainable_params"],
        "total_params": result["total_params"],
        "best_epoch": result["best_epoch"],
        "val_acc": result["val_metrics"].get("acc"),
        "val_macro_f1": result["val_metrics"].get("macro_f1"),
        "test_acc": result["test_metrics"].get("acc"),
        "test_macro_f1": result["test_metrics"].get("macro_f1"),
        "artifact_paths": result["artifact_paths"],
    }

    print("\n=== fine_tune summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()