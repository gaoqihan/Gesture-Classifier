from __future__ import annotations

import argparse
import json

from gesture_classifier.add_function_lib import add_function


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
        description="Add a new gesture class/function to an existing gesture classifier."
    )

    parser.add_argument(
        "--data_root",
        required=True,
        type=str,
        help="Root folder containing data arranged by label folders.",
    )
    parser.add_argument(
        "--new_class_name",
        required=True,
        type=str,
        help="Name of the new gesture class to add.",
    )

    parser.add_argument(
        "--new_class_train_samples",
        default=None,
        type=int,
        help="Maximum number of training samples to keep for the new class in the training split.",
    )
    parser.add_argument(
        "--init_mode",
        default="expand_from_checkpoint",
        choices=["expand_from_checkpoint", "scratch"],
        help="Whether to expand an existing checkpoint or train from scratch.",
    )
    parser.add_argument(
        "--previous_run_dir",
        default=None,
        type=str,
        help="Previous run directory containing config.json and best_model.pt. Required for expand_from_checkpoint.",
    )

    parser.add_argument(
        "--logger_root",
        default="./logger_add_function",
        type=str,
        help="Root folder where run logs and checkpoints will be saved.",
    )
    parser.add_argument(
        "--run_note",
        default="add_new_class",
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
        "--epochs",
        default=400,
        type=int,
        help="Training epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        default=1e-4,
        type=float,
        help="Weight decay.",
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size.",
    )
    parser.add_argument(
        "--num_workers",
        default=0,
        type=int,
        help="DataLoader num_workers.",
    )

    parser.add_argument(
        "--use_class_weights",
        default=True,
        type=str2bool,
        help="Whether to use inverse-frequency class weights.",
    )
    parser.add_argument(
        "--use_lr_scheduler",
        default=True,
        type=str2bool,
        help="Whether to use ReduceLROnPlateau scheduler.",
    )

    parser.add_argument(
        "--early_stopping_patience",
        default=10,
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
        help="If true and using expand_from_checkpoint, only train classifier head.",
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

    result = add_function(
        data_root=args.data_root,
        new_class_name=args.new_class_name,
        new_class_train_samples=args.new_class_train_samples,
        init_mode=args.init_mode,
        previous_run_dir=args.previous_run_dir,
        logger_root=args.logger_root,
        run_note=args.run_note,
        random_seed=args.random_seed,
        stratify=args.stratify,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_class_weights=args.use_class_weights,
        use_lr_scheduler=args.use_lr_scheduler,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        freeze_backbone=args.freeze_backbone,
        device=args.device,
    )

    summary = {
        "run_dir": result["run_dir"],
        "device": result["device"],
        "labels": result["labels"],
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

    print("\n=== add_function summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()