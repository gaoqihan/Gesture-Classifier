from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class SmallFrameCNN(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B*T, 3, H, W]
        returns: [B*T, out_dim]
        """
        x = self.features(x)
        x = x.flatten(1)
        x = self.proj(x)
        return x


class VideoEncoder(nn.Module):
    def __init__(self, video_feature_dim: int = 64):
        super().__init__()
        self.frame_cnn = SmallFrameCNN(out_dim=video_feature_dim)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        video: [B, T, C, H, W]
        returns: [B, T, video_feature_dim]
        """
        b, t, c, h, w = video.shape
        x = video.reshape(b * t, c, h, w)
        x = self.frame_cnn(x)
        x = x.reshape(b, t, -1)
        return x


class PoseMLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, input_dim]
        returns: [B, T, out_dim]
        """
        b, t, d = x.shape
        x = x.reshape(b * t, d)
        x = self.net(x)
        x = x.reshape(b, t, -1)
        return x


class BodyEncoder(nn.Module):
    def __init__(
        self,
        num_landmarks: int = 17,
        coord_dim: int = 2,
        hidden_dim: int = 128,
        out_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.coord_dim = coord_dim
        self.encoder = PoseMLPEncoder(
            input_dim=num_landmarks * coord_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
        )

    def forward(self, body: torch.Tensor) -> torch.Tensor:
        """
        body: [B, T, K, 2]
        returns: [B, T, out_dim]
        """
        b, t, k, d = body.shape
        x = body.reshape(b, t, k * d)
        return self.encoder(x)


class HandEncoder(nn.Module):
    def __init__(
        self,
        num_landmarks: int = 21,
        coord_dim: int = 2,
        hidden_dim: int = 128,
        out_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.coord_dim = coord_dim
        self.encoder = PoseMLPEncoder(
            input_dim=num_landmarks * coord_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
        )

    def forward(self, hand: torch.Tensor) -> torch.Tensor:
        """
        hand: [B, T, K, 2]
        returns: [B, T, out_dim]
        """
        b, t, k, d = hand.shape
        x = hand.reshape(b, t, k * d)
        return self.encoder(x)


class MultiModalEarlyFusionEncoder(nn.Module):
    def __init__(
        self,
        video_input_enabled: bool = True,
        body_pose_input_enabled: bool = True,
        hand_pose_input_enabled: bool = True,
        video_feature_dim: int = 64,
        body_feature_dim: int = 32,
        hand_feature_dim: int = 32,
        body_num_landmarks: int = 17,
        hand_num_landmarks: int = 21,
        pose_hidden_dim: int = 128,
        pose_dropout: float = 0.1,
    ):
        super().__init__()

        self.video_input_enabled = video_input_enabled
        self.body_pose_input_enabled = body_pose_input_enabled
        self.hand_pose_input_enabled = hand_pose_input_enabled

        if not (
            self.video_input_enabled
            or self.body_pose_input_enabled
            or self.hand_pose_input_enabled
        ):
            raise ValueError("At least one modality must be enabled.")

        self.video_encoder = None
        self.body_encoder = None
        self.hand_encoder = None

        fusion_dim = 0

        if self.video_input_enabled:
            self.video_encoder = VideoEncoder(video_feature_dim=video_feature_dim)
            fusion_dim += video_feature_dim

        if self.body_pose_input_enabled:
            self.body_encoder = BodyEncoder(
                num_landmarks=body_num_landmarks,
                coord_dim=2,
                hidden_dim=pose_hidden_dim,
                out_dim=body_feature_dim,
                dropout=pose_dropout,
            )
            fusion_dim += body_feature_dim

        if self.hand_pose_input_enabled:
            self.hand_encoder = HandEncoder(
                num_landmarks=hand_num_landmarks,
                coord_dim=2,
                hidden_dim=pose_hidden_dim,
                out_dim=hand_feature_dim,
                dropout=pose_dropout,
            )
            fusion_dim += hand_feature_dim

        self.fusion_dim = fusion_dim

    def forward(
        self,
        video: Optional[torch.Tensor] = None,
        body: Optional[torch.Tensor] = None,
        hand: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        returns fused sequence features: [B, T, fusion_dim]
        """
        features = []

        if self.video_input_enabled:
            if video is None:
                raise ValueError("video modality is enabled but video input is None")
            features.append(self.video_encoder(video))

        if self.body_pose_input_enabled:
            if body is None:
                raise ValueError("body modality is enabled but body input is None")
            features.append(self.body_encoder(body))

        if self.hand_pose_input_enabled:
            if hand is None:
                raise ValueError("hand modality is enabled but hand input is None")
            features.append(self.hand_encoder(hand))

        if len(features) == 1:
            return features[0]

        return torch.cat(features, dim=-1)


class GestureLSTMClassifier(nn.Module):
    def __init__(
        self,
        encoder: MultiModalEarlyFusionEncoder,
        num_classes: int,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 1,
        bidirectional: bool = False,
        classifier_dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = encoder
        self.bidirectional = bidirectional
        self.lstm_hidden_dim = lstm_hidden_dim

        self.lstm = nn.LSTM(
            input_size=self.encoder.fusion_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_out_dim = lstm_hidden_dim * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(lstm_out_dim, num_classes),
        )

    def forward(
        self,
        video: Optional[torch.Tensor] = None,
        body: Optional[torch.Tensor] = None,
        hand: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        returns logits: [B, num_classes]
        """
        x = self.encoder(video=video, body=body, hand=hand)  # [B, T, F]
        x, _ = self.lstm(x)  # [B, T, H]
        x = x[:, -1, :]
        logits = self.classifier(x)
        return logits


class GestureCNNClassifier(nn.Module):
    def __init__(
        self,
        encoder: MultiModalEarlyFusionEncoder,
        num_classes: int,
        temporal_hidden_dim: int = 128,
        classifier_dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = encoder

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=self.encoder.fusion_dim,
                out_channels=temporal_hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(temporal_hidden_dim),

            nn.Conv1d(
                in_channels=temporal_hidden_dim,
                out_channels=temporal_hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(temporal_hidden_dim),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(temporal_hidden_dim, num_classes),
        )

    def forward(
        self,
        video: Optional[torch.Tensor] = None,
        body: Optional[torch.Tensor] = None,
        hand: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        returns logits: [B, num_classes]
        """
        x = self.encoder(video=video, body=body, hand=hand)  # [B, T, F]
        x = x.transpose(1, 2)  # [B, F, T]
        x = self.temporal_conv(x)  # [B, H, 1]
        x = x.squeeze(-1)  # [B, H]
        logits = self.classifier(x)
        return logits


def infer_body_num_landmarks_from_config(cfg: Dict) -> int:
    body_landmarks = cfg.get("BODY_LANDMARK_USED", None)
    if body_landmarks is None:
        return 17
    return len(body_landmarks)


def build_encoder_from_config(cfg: Dict) -> MultiModalEarlyFusionEncoder:
    video_input_enabled = bool(cfg.get("VIDEO_INPUT_ENABLED", True))
    body_pose_input_enabled = bool(cfg.get("BODY_POSE_INPUT_ENABLED", True))
    hand_pose_input_enabled = bool(cfg.get("HAND_POSE_INPUT_ENABLED", True))

    video_feature_dim = int(cfg.get("VIDEO_FEATURE_DIM", 64))
    body_feature_dim = int(cfg.get("BODY_FEATURE_DIM", 32))
    hand_feature_dim = int(cfg.get("HAND_FEATURE_DIM", 32))
    pose_hidden_dim = int(cfg.get("POSE_HIDDEN_DIM", 128))
    pose_dropout = float(cfg.get("POSE_DROPOUT", 0.1))

    body_num_landmarks = infer_body_num_landmarks_from_config(cfg)
    hand_num_landmarks = int(cfg.get("HAND_NUM_LANDMARKS", 21))

    return MultiModalEarlyFusionEncoder(
        video_input_enabled=video_input_enabled,
        body_pose_input_enabled=body_pose_input_enabled,
        hand_pose_input_enabled=hand_pose_input_enabled,
        video_feature_dim=video_feature_dim,
        body_feature_dim=body_feature_dim,
        hand_feature_dim=hand_feature_dim,
        body_num_landmarks=body_num_landmarks,
        hand_num_landmarks=hand_num_landmarks,
        pose_hidden_dim=pose_hidden_dim,
        pose_dropout=pose_dropout,
    )


def build_model_from_config_with_num_classes(
    cfg: Dict,
    num_classes: int,
) -> nn.Module:
    model_type = str(cfg.get("MODEL_TYPE", "CNN")).upper()

    encoder = build_encoder_from_config(cfg)

    if model_type == "LSTM":
        return GestureLSTMClassifier(
            encoder=encoder,
            num_classes=num_classes,
            lstm_hidden_dim=int(cfg.get("LSTM_HIDDEN_DIM", 128)),
            lstm_num_layers=int(cfg.get("LSTM_NUM_LAYERS", 1)),
            bidirectional=bool(cfg.get("LSTM_BIDIRECTIONAL", False)),
            classifier_dropout=float(cfg.get("CLASSIFIER_DROPOUT", 0.2)),
        )

    if model_type == "CNN":
        return GestureCNNClassifier(
            encoder=encoder,
            num_classes=num_classes,
            temporal_hidden_dim=int(cfg.get("TEMPORAL_HIDDEN_DIM", 128)),
            classifier_dropout=float(cfg.get("CLASSIFIER_DROPOUT", 0.2)),
        )

    raise ValueError(f"Unsupported MODEL_TYPE: {model_type}")


def expand_final_linear_layer(
    model: nn.Module,
    old_num_classes: int,
    new_num_classes: int,
) -> nn.Module:
    """
    Expands the final linear classifier layer while preserving old weights.

    Supported:
    - GestureLSTMClassifier
    - GestureCNNClassifier
    """
    if new_num_classes <= old_num_classes:
        raise ValueError(
            f"new_num_classes ({new_num_classes}) must be > old_num_classes ({old_num_classes})"
        )

    if not hasattr(model, "classifier"):
        raise ValueError("Model has no attribute 'classifier'")

    if not isinstance(model.classifier, nn.Sequential):
        raise ValueError("Expected model.classifier to be nn.Sequential")

    if len(model.classifier) == 0:
        raise ValueError("model.classifier is empty")

    last_layer = model.classifier[-1]
    if not isinstance(last_layer, nn.Linear):
        raise ValueError("Last classifier layer is not nn.Linear")

    old_linear: nn.Linear = last_layer
    in_features = old_linear.in_features
    device = old_linear.weight.device
    dtype = old_linear.weight.dtype

    new_linear = nn.Linear(in_features, new_num_classes).to(device=device, dtype=dtype)

    with torch.no_grad():
        new_linear.weight[:old_num_classes].copy_(old_linear.weight)
        new_linear.bias[:old_num_classes].copy_(old_linear.bias)

    model.classifier[-1] = new_linear
    return model


def freeze_encoder_except_classifier(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True


def freeze_all_except_last_n_modules(model: nn.Module, last_n_modules: int = 1) -> None:
    """
    Generic helper if later you want partial unfreezing.
    """
    modules = list(model.children())

    for param in model.parameters():
        param.requires_grad = False

    if last_n_modules <= 0:
        return

    for module in modules[-last_n_modules:]:
        for param in module.parameters():
            param.requires_grad = True


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_model_structure_string(model: nn.Module) -> str:
    return str(model)