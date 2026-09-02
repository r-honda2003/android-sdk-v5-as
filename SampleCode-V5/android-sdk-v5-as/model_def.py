import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class MapMatchingPolicy(nn.Module):
    def __init__(self, history_len: int, freeze_backbone: bool = True):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V2
        backbone = mobilenet_v2(weights=weights)
        self.features = backbone.features
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        history_dim = history_len * 4  # 4 channels: stick_left_v, stick_left_h, stick_right_v, stick_right_h
        # 1280 (curr_img) + 1280 (ref_img) + history_dim
        self.head = nn.Sequential(
            nn.Linear(1280 * 2 + history_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 4),
            nn.Tanh(),
        )

    def forward(self, x, ref_x, history):
        # 両方の画像から特徴抽出 (重みは共有)
        feat = self.flatten(self.pool(self.features(x)))
        ref_feat = self.flatten(self.pool(self.features(ref_x)))

        combined = torch.cat([feat, ref_feat, history], dim=1)
        return self.head(combined)
