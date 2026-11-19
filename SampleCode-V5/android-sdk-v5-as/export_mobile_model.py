import argparse
import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class MapMatchingPolicy(nn.Module):
    def __init__(self, history_len: int = 5, freeze_backbone: bool = True):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V2
        backbone = mobilenet_v2(weights=weights)
        self.features = backbone.features
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        history_dim = history_len * 4
        self.head = nn.Sequential(
            nn.Linear(1280 * 2 + history_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 4),
            nn.Tanh(),
        )

    def forward(self, x, ref_x, history):
        feat = self.flatten(self.pool(self.features(x)))
        ref_feat = self.flatten(self.pool(self.features(ref_x)))
        combined = torch.cat([feat, ref_feat, history], dim=1)
        return self.head(combined)


class BaselinePolicy(nn.Module):
    """Pure E2E Baseline Policy: Single camera image + stick history -> 4 stick channels"""
    def __init__(self, history_len: int = 5, freeze_backbone: bool = True):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V2
        backbone = mobilenet_v2(weights=weights)
        self.features = backbone.features
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        history_dim = history_len * 4
        self.head = nn.Sequential(
            nn.Linear(1280 + history_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 4),
            nn.Tanh(),
        )

    def forward(self, x, history):
        feat = self.flatten(self.pool(self.features(x)))
        combined = torch.cat([feat, history], dim=1)
        return self.head(combined)


def export_model(checkpoint_path="map_matching_model.pt", output_path="map_matching_model_android.pt", model_type="auto"):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    history_len = checkpoint.get("history_len", 5)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Auto detect baseline vs map_matching if not specified
    if model_type == "auto":
        is_baseline = "baseline" in checkpoint_path.lower() or "head.0.weight" in state_dict and state_dict["head.0.weight"].shape[1] < 2000
        model_type = "baseline" if is_baseline else "map_matching"

    example_x = torch.randn(1, 3, 224, 224)
    example_hist = torch.randn(1, history_len * 4)

    if model_type == "baseline":
        print(f"Exporting Baseline Pure E2E Model (Single Image + History) from {checkpoint_path}...")
        model = BaselinePolicy(history_len=history_len, freeze_backbone=True)
        model.load_state_dict(state_dict)
        model.eval()
        traced = torch.jit.trace(model, (example_x, example_hist))
    else:
        print(f"Exporting Map Matching Model (Dual Image + History) from {checkpoint_path}...")
        model = MapMatchingPolicy(history_len=history_len, freeze_backbone=True)
        model.load_state_dict(state_dict)
        model.eval()
        example_ref = torch.randn(1, 3, 224, 224)
        traced = torch.jit.trace(model, (example_x, example_ref, example_hist))

    traced.save(output_path)
    print(f"Successfully exported mobile TorchScript model ({model_type}) to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch state_dict to TorchScript for PyTorch Mobile Android.")
    parser.add_argument("--model", default="map_matching_model.pt", help="Input state_dict model checkpoint")
    parser.add_argument("--output", default="map_matching_model_android.pt", help="Output TorchScript model for Android")
    parser.add_argument("--type", choices=["auto", "map_matching", "baseline"], default="auto", help="Model architecture type")
    args = parser.parse_args()

    export_model(args.model, args.output, args.type)

if __name__ == "__main__":
    main()
