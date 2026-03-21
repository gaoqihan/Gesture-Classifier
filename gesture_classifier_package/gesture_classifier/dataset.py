from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset


def build_label_mappings(labels: Sequence[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    label_to_idx = {name: i for i, name in enumerate(labels)}
    idx_to_label = {i: name for name, i in label_to_idx.items()}
    return label_to_idx, idx_to_label


def apply_body_landmark_selection(
    body_np: np.ndarray,
    landmark_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    if landmark_indices is None:
        return body_np
    return body_np[:, landmark_indices, :]


def check_video_readable(video_path: str) -> Tuple[bool, Optional[str]]:
    if not os.path.isfile(video_path):
        return False, "video file missing"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "video cannot be opened"

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return False, "video has no readable frames"

    return True, None


def check_npy_shape(
    npy_path: str,
    expected_num_landmarks: int,
    name: str,
) -> Tuple[bool, Optional[Tuple[int, ...]], Optional[str]]:
    """
    Expected shape: [T, expected_num_landmarks, 2]
    """
    if not os.path.isfile(npy_path):
        return False, None, f"{name} file missing"

    try:
        arr = np.load(npy_path)
    except Exception as e:
        return False, None, f"{name} npy load failed: {e}"

    if arr.ndim != 3:
        return False, arr.shape, f"{name} ndim should be 3, got {arr.ndim}"

    if arr.shape[1] != expected_num_landmarks:
        return (
            False,
            arr.shape,
            f"{name} landmark count should be {expected_num_landmarks}, got {arr.shape[1]}",
        )

    if arr.shape[2] != 2:
        return False, arr.shape, f"{name} last dim should be 2, got {arr.shape[2]}"

    if arr.shape[0] <= 0:
        return False, arr.shape, f"{name} has zero length"

    return True, arr.shape, None


def load_video_cv2(
    video_path: str,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
) -> torch.Tensor:
    """
    Returns:
        video tensor of shape [T, C, H, W]
    """
    cap = cv2.VideoCapture(video_path)
    frames: List[torch.Tensor] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size)
        frame = torch.from_numpy(frame).float()  # [H, W, C]

        if normalize:
            frame = frame / 255.0

        frame = frame.permute(2, 0, 1)  # [C, H, W]
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Could not read frames from video: {video_path}")

    return torch.stack(frames, dim=0)  # [T, C, H, W]


def pad_or_truncate_tensor(
    x: torch.Tensor,
    target_len: int,
    truncate_if_longer: bool = True,
) -> torch.Tensor:
    """
    Input shape: [T, ...]
    Output shape: [target_len, ...] if truncation/padding is applied.
    """
    cur_len = x.shape[0]

    if cur_len == target_len:
        return x

    if cur_len > target_len:
        return x[:target_len] if truncate_if_longer else x

    pad_shape = (target_len - cur_len,) + tuple(x.shape[1:])
    pad_tensor = torch.zeros(pad_shape, dtype=x.dtype)
    return torch.cat([x, pad_tensor], dim=0)


class GestureEpisodeDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        labels: Sequence[str],
        video_input_enabled: bool = True,
        body_pose_input_enabled: bool = True,
        hand_pose_input_enabled: bool = True,
        body_landmark_used: Optional[Sequence[int]] = None,
        video_size: Tuple[int, int] = (224, 224),
        video_normalize: bool = True,
        video_pad_length: Optional[int] = None,
        body_pad_length: Optional[int] = None,
        hand_pad_length: Optional[int] = None,
        truncate_if_longer: bool = True,
        require_all_three_components: bool = True,
        verbose_screening: bool = True,
    ) -> None:
        self.data_root = data_root
        self.labels = list(labels)

        self.video_input_enabled = video_input_enabled
        self.body_pose_input_enabled = body_pose_input_enabled
        self.hand_pose_input_enabled = hand_pose_input_enabled

        self.body_landmark_used = list(body_landmark_used) if body_landmark_used is not None else None
        self.video_size = tuple(video_size)
        self.video_normalize = video_normalize

        self.video_pad_length = video_pad_length
        self.body_pad_length = body_pad_length
        self.hand_pad_length = hand_pad_length
        self.truncate_if_longer = truncate_if_longer

        self.require_all_three_components = require_all_three_components
        self.verbose_screening = verbose_screening

        self.label_to_idx, self.idx_to_label = build_label_mappings(self.labels)

        if not (
            self.video_input_enabled
            or self.body_pose_input_enabled
            or self.hand_pose_input_enabled
        ):
            raise ValueError("At least one input modality must be enabled.")

        self.samples = self._build_index()

        if len(self.samples) == 0:
            raise ValueError("No valid samples found after screening.")

    def _screen_one_sample(
        self,
        body_path: str,
        hand_path: str,
        video_path: str,
    ) -> Tuple[bool, Optional[List[str]]]:
        reasons: List[str] = []

        if self.require_all_three_components:
            if not os.path.isfile(body_path):
                reasons.append("missing body file")
            if not os.path.isfile(hand_path):
                reasons.append("missing hand file")
            if not os.path.isfile(video_path):
                reasons.append("missing video file")
            if reasons:
                return False, reasons
        else:
            if self.body_pose_input_enabled and not os.path.isfile(body_path):
                reasons.append("missing body file")
            if self.hand_pose_input_enabled and not os.path.isfile(hand_path):
                reasons.append("missing hand file")
            if self.video_input_enabled and not os.path.isfile(video_path):
                reasons.append("missing video file")
            if reasons:
                return False, reasons

        check_body = self.require_all_three_components or self.body_pose_input_enabled
        check_hand = self.require_all_three_components or self.hand_pose_input_enabled
        check_video = self.require_all_three_components or self.video_input_enabled

        if check_body:
            ok, shape, reason = check_npy_shape(body_path, 17, "body")
            if not ok:
                reasons.append(reason if reason is not None else f"invalid body shape: {shape}")

        if check_hand:
            ok, shape, reason = check_npy_shape(hand_path, 21, "hand")
            if not ok:
                reasons.append(reason if reason is not None else f"invalid hand shape: {shape}")

        if check_video:
            ok, reason = check_video_readable(video_path)
            if not ok:
                reasons.append(reason or "invalid video")

        if reasons:
            return False, reasons

        return True, None

    def _build_index(self) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        skipped: List[Tuple[str, str]] = []

        for class_name in self.labels:
            class_dir = os.path.join(self.data_root, class_name)
            if not os.path.isdir(class_dir):
                skipped.append((class_name, "class folder missing"))
                continue

            all_session_prefixes = set()

            for path in glob.glob(os.path.join(class_dir, "session_*_body.npy")):
                all_session_prefixes.add(os.path.basename(path).replace("_body.npy", ""))

            for path in glob.glob(os.path.join(class_dir, "session_*_hand.npy")):
                all_session_prefixes.add(os.path.basename(path).replace("_hand.npy", ""))

            for path in glob.glob(os.path.join(class_dir, "session_*.mp4")):
                all_session_prefixes.add(os.path.basename(path).replace(".mp4", ""))

            for base in sorted(all_session_prefixes):
                body_path = os.path.join(class_dir, f"{base}_body.npy")
                hand_path = os.path.join(class_dir, f"{base}_hand.npy")
                video_path = os.path.join(class_dir, f"{base}.mp4")

                ok, reasons = self._screen_one_sample(body_path, hand_path, video_path)

                if not ok:
                    skipped.append((f"{class_name}/{base}", "; ".join(reasons or [])))
                    continue

                samples.append(
                    {
                        "label_name": class_name,
                        "label_idx": self.label_to_idx[class_name],
                        "body_path": body_path,
                        "hand_path": hand_path,
                        "video_path": video_path,
                        "session_id": base,
                    }
                )

        if self.verbose_screening:
            print(f"Valid samples: {len(samples)}")
            print(f"Skipped samples: {len(skipped)}")
            if len(skipped) > 0:
                print("\nSome skipped samples:")
                for item, reason in skipped[:30]:
                    print(f"  - {item}: {reason}")
                if len(skipped) > 30:
                    print(f"  ... and {len(skipped) - 30} more")

        self.skipped_samples = skipped
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample_info = self.samples[idx]
        outputs: List[Any] = []

        body_input_shape = None

        if self.video_input_enabled:
            video = load_video_cv2(
                sample_info["video_path"],
                target_size=self.video_size,
                normalize=self.video_normalize,
            )  # [T, C, H, W]

            if self.video_pad_length is not None:
                video = pad_or_truncate_tensor(
                    video,
                    self.video_pad_length,
                    truncate_if_longer=self.truncate_if_longer,
                )

            outputs.append(video)

        if self.body_pose_input_enabled:
            body = np.load(sample_info["body_path"]).astype(np.float32)  # [T,17,2]
            body = apply_body_landmark_selection(body, self.body_landmark_used)
            body = torch.from_numpy(body)  # [T,K,2]

            if self.body_pad_length is not None:
                body = pad_or_truncate_tensor(
                    body,
                    self.body_pad_length,
                    truncate_if_longer=self.truncate_if_longer,
                )

            body_input_shape = tuple(body.shape)
            outputs.append(body)

        if self.hand_pose_input_enabled:
            hand = np.load(sample_info["hand_path"]).astype(np.float32)  # [T,21,2]
            hand = torch.from_numpy(hand)

            if self.hand_pad_length is not None:
                hand = pad_or_truncate_tensor(
                    hand,
                    self.hand_pad_length,
                    truncate_if_longer=self.truncate_if_longer,
                )

            outputs.append(hand)

        if self.body_pose_input_enabled:
            outputs.append(body_input_shape)

        label = torch.tensor(sample_info["label_idx"], dtype=torch.long)
        return tuple(outputs), label, sample_info["session_id"]


def gesture_collate_fn(batch):
    inputs_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch], dim=0)
    session_ids = [item[2] for item in batch]
    return inputs_list, labels, session_ids


def gesture_collate_fn_stacked(batch):
    labels = torch.stack([item[1] for item in batch], dim=0)
    session_ids = [item[2] for item in batch]

    sample0 = batch[0][0]
    num_items = len(sample0)

    collated_inputs = []

    for item_idx in range(num_items):
        values = [sample[0][item_idx] for sample in batch]

        if torch.is_tensor(values[0]):
            collated_inputs.append(torch.stack(values, dim=0))
        else:
            collated_inputs.append(values)

    return tuple(collated_inputs), labels, session_ids


def make_train_val_test_subsets(
    dataset: GestureEpisodeDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    stratify_split: bool = True,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8

    indices = list(range(len(dataset)))
    labels = [dataset.samples[i]["label_idx"] for i in indices]

    stratify_labels = labels if stratify_split else None

    train_indices, temp_indices = train_test_split(
        indices,
        test_size=(1.0 - train_ratio),
        random_state=random_seed,
        stratify=stratify_labels,
    )

    temp_labels = [dataset.samples[i]["label_idx"] for i in temp_indices]
    temp_stratify = temp_labels if stratify_split else None

    val_portion_of_temp = val_ratio / (val_ratio + test_ratio)

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1.0 - val_portion_of_temp),
        random_state=random_seed,
        stratify=temp_stratify,
    )

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    return train_subset, val_subset, test_subset, train_indices, val_indices, test_indices


def make_train_val_test_subsets_for_new_class(
    dataset: GestureEpisodeDataset,
    labels: Sequence[str],
    new_class_name: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    stratify: bool = True,
    new_class_train_samples: Optional[int] = None,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8

    indices = list(range(len(dataset)))
    y = [dataset.samples[i]["label_idx"] for i in indices]
    stratify_y = y if stratify else None

    train_indices, temp_indices = train_test_split(
        indices,
        test_size=(1.0 - train_ratio),
        random_state=random_seed,
        stratify=stratify_y,
    )

    temp_y = [dataset.samples[i]["label_idx"] for i in temp_indices]
    temp_stratify_y = temp_y if stratify else None

    val_portion_of_temp = val_ratio / (val_ratio + test_ratio)

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1.0 - val_portion_of_temp),
        random_state=random_seed,
        stratify=temp_stratify_y,
    )

    if new_class_train_samples is not None:
        label_to_idx = {name: i for i, name in enumerate(labels)}
        new_class_idx = label_to_idx[new_class_name]

        new_class_train_idx = [
            i for i in train_indices if dataset.samples[i]["label_idx"] == new_class_idx
        ]
        other_train_idx = [
            i for i in train_indices if dataset.samples[i]["label_idx"] != new_class_idx
        ]

        if len(new_class_train_idx) == 0:
            raise ValueError(f"No training samples found for new class '{new_class_name}'")

        rng = np.random.RandomState(random_seed)
        rng.shuffle(new_class_train_idx)

        kept_new_class_train_idx = new_class_train_idx[:new_class_train_samples]
        train_indices = other_train_idx + kept_new_class_train_idx
        rng.shuffle(train_indices)

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    return train_subset, val_subset, test_subset, train_indices, val_indices, test_indices


def summarize_subset(subset: Subset, dataset: GestureEpisodeDataset, name: str = "subset") -> None:
    counts: Dict[str, int] = {}
    for idx in subset.indices:
        label_name = dataset.samples[idx]["label_name"]
        counts[label_name] = counts.get(label_name, 0) + 1

    print(f"{name}: {len(subset)} samples")
    for label in dataset.labels:
        print(f"  {label}: {counts.get(label, 0)}")


def summarize_subset_generic(
    subset: Subset,
    base_dataset: GestureEpisodeDataset,
    labels: Sequence[str],
    name: str = "subset",
) -> None:
    counts: Dict[str, int] = {}
    for idx in subset.indices:
        label_name = base_dataset.samples[idx]["label_name"]
        counts[label_name] = counts.get(label_name, 0) + 1

    print(f"{name}: {len(subset)} samples")
    for label in labels:
        print(f"  {label}: {counts.get(label, 0)}")