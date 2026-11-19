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

def export_model(checkpoint_path="map_matching_model.pt", output_path="map_matching_model_android.pt"):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    history_len = checkpoint.get("history_len", 5)

    model = MapMatchingPolicy(history_len=history_len, freeze_backbone=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    example_x = torch.randn(1, 3, 224, 224)
    example_ref = torch.randn(1, 3, 224, 224)
    example_hist = torch.randn(1, history_len * 4)

    traced = torch.jit.trace(model, (example_x, example_ref, example_hist))
    traced.save(output_path)

    print(f"Successfully exported mobile TorchScript model to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch state_dict to TorchScript for PyTorch Mobile Android.")
    parser.add_argument("--model", default="map_matching_model.pt", help="Input state_dict model checkpoint")
    parser.add_argument("--output", default="map_matching_model_android.pt", help="Output TorchScript model for Android")
    args = parser.parse_args()

    export_model(args.model, args.output)

if __name__ == "__main__":
    main()
