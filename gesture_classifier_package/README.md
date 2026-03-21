# Gesture Classifier (ROS2 + PyTorch)

A multimodal gesture classification framework supporting:

- RGB video
- Body pose (17 keypoints)
- Hand pose (21 keypoints)

Features:
- Fine-tuning existing models
- Adding new gesture classes (incremental learning)
- Integration with ROS2 (colcon) for CLI workflows

---

## 📦 Package structure

```
gesture_classifier/
├── dataset.py
├── models.py
├── train_utils.py
├── logger_utils.py
├── io_utils.py
├── fine_tune_lib.py
├── add_function_lib.py
├── fine_tune.py
└── add_function.py
```

---

## 📁 Dataset format

```
data_root/
├── go/
│   ├── session_0001.mp4
│   ├── session_0001_body.npy   # [T,17,2]
│   └── session_0001_hand.npy   # [T,21,2]
```

---

## ⚙️ Build

```bash
cd ~/ros2_ws
colcon build --packages-select gesture_classifier
source install/setup.bash
```

---

## 🧠 Model

Supported architectures:
- CNN (recommended)
- LSTM (sensitive to padding)

---

## 🚀 Commands

fine_tune:
```
ros2 run gesture_classifier fine_tune [ARGS]
```

add_function:
```
ros2 run gesture_classifier add_function [ARGS]
```

---

## fine_tune — required arguments

| Argument | Description |
|----------|-------------|
| --finetuning_data_root | Root folder containing fine-tuning data arranged by label folders. |
| --pretrained_run_dir    | Run directory containing config.json and best_model.pt from a previous run. |
| --finetune_num_samples | Maximum number of training samples to use from the fine-tuning split. |

---

## Training defaults

These defaults are taken from `gesture_classifier/fine_tune.py` (so they match the CLI parser):

| Parameter | Default |
|-----------|---------|
| --finetune_epochs | 200 |
| --finetune_learning_rate | 1e-2 |
| --finetune_weight_decay | 1e-4 |
| --finetune_batch_size | 4 |
| --finetune_num_workers | 0 |

---

## Behavior flags

| Parameter | Default |
|-----------|---------|
| --freeze_backbone | True |
| --finetune_use_class_weights | False |
| --finetune_use_lr_scheduler | False |
| --inference_only | False |

---

## Early stopping defaults

| Parameter | Default |
|-----------|---------|
| --early_stopping_patience | 20 |
| --early_stopping_min_delta | 1e-4 |

---

## System / device

| Parameter | Example |
|-----------|---------|
| --device | cuda / cuda:0 / cpu (default: auto-detect when not provided) |

---

## ✅ Example (fine-tune)

This example uses the argument names from `fine_tune.py`. Adjust paths and values as needed.

```bash
ros2 run gesture_classifier fine_tune \
	--finetuning_data_root /home/qihan/data_finetune \
	--pretrained_run_dir /home/qihan/logger/run_001 \
	--finetune_num_samples 100 \
	--finetune_epochs 200 \
	--finetune_learning_rate 1e-2 \
	--finetune_weight_decay 1e-4 \
	--finetune_batch_size 4 \
	--freeze_backbone true \
	--logger_root ./logger_finetuning \
	--device cuda:0
```

---

## 📊 Outputs

Each run creates a logger folder:

```
logger_root/
└── timestamp__MODEL__...
		├── config.json
		├── history.json
		├── metrics.json
		├── best_model.pt
		├── training_curve.png
		└── confusion_matrix_*.csv
```

---

## 📈 Meta tracking

`logger_root/meta_runs.csv` — contains hyperparameters, performance, and run path.

---

## Notes

- CNN → stable
- LSTM → sensitive to padding

Few-shot (small samples &lt; 20):
- Unstable; overfitting likely

Freeze strategy:
- `freeze_backbone = True` → use for small data
- `freeze_backbone = False` → use for large data

---

## Workflow

1. Train base model
2. Fine-tune per user
3. Add new gestures

---

## Summary

| Task | Command |
|------|---------|
| Adapt model | `fine_tune` |
| Add gesture | `add_function` |

---

## add_function — defaults & example

The `add_function` CLI defaults are chosen to match the notebook configuration used for adding a new class in experiments.

Defaults:

| Parameter | Default |
|-----------|---------|
| --epochs | 400 |
| --learning_rate | 1e-4 |
| --weight_decay | 1e-4 |
| --batch_size | 16 |
| --num_workers | 0 |
| --use_class_weights | True |
| --use_lr_scheduler | True |
| --early_stopping_patience | 10 |
| --early_stopping_min_delta | 1e-4 |

Example (add a new class `sit` by expanding from a checkpoint):

```bash
ros2 run gesture_classifier add_function \
	--data_root /home/qihan/data_full \
	--new_class_name sit \
	--init_mode expand_from_checkpoint \
	--previous_run_dir /home/qihan/logger/run_001 \
	--new_class_train_samples 20 \
	--epochs 400 \
	--learning_rate 1e-4 \
	--batch_size 16 \
	--use_class_weights true \
	--use_lr_scheduler true \
	--freeze_backbone true \
	--logger_root ./logger_add_function \
	--device cuda:0
```
