"""
evaluate_map_matching.py

学習済みの map_matching_model.pt を使い、validationセッションに対する予測誤差を確認する。
「今の画像」＋「基準画像（地図）」をモデルに与えて推論を行う。

使い方:
    python3 evaluate_map_matching.py --telemetry matched_telemetry.csv --images_dir images_20260827_184612 \
        --model map_matching_model.pt --val_session_id 6
"""

import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from train_map_matching import MapMatchingPolicy, MapMatchingDataset, load_dataset_df, STICK_MAX, STICK_COLS


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained map-matching model.")
    parser.add_argument("--telemetry", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--model", required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--val_session_id", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_examples", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint = torch.load(args.model, map_location="cpu")
    history_len = checkpoint["history_len"]
    print(f"モデルのhistory_len: {history_len}")

    df = load_dataset_df(args.telemetry, history_len)

    available_sessions = sorted(df["session_id"].unique())
    val_session_id = args.val_session_id if args.val_session_id != -1 else max(available_sessions)
    val_df = df[df["session_id"] == val_session_id].reset_index(drop=True)
    print(f"評価対象: session {val_session_id} ({len(val_df)}枚)")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_ds = MapMatchingDataset(val_df, args.images_dir, transform, history_len)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MapMatchingPolicy(history_len=history_len, freeze_backbone=True).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, ref_images, history, targets in val_loader:
            images, ref_images, history = images.to(device), ref_images.to(device), history.to(device)
            preds = model(images, ref_images, history).cpu()
            all_preds.append(preds)
            all_targets.append(targets)

    preds = torch.cat(all_preds, dim=0) * STICK_MAX
    targets = torch.cat(all_targets, dim=0) * STICK_MAX

    abs_err = (preds - targets).abs()
    mae_per_axis = abs_err.mean(dim=0)

    print("\n=== スティック値の単位(-660〜660)での平均誤差(MAE) ===")
    for i, col in enumerate(STICK_COLS):
        print(f"{col:>16}: {mae_per_axis[i]:7.1f}")
    print(f"{'全軸平均':>16}: {mae_per_axis.mean():7.1f}")

    print(f"\n=== サンプル比較(先頭{args.num_examples}件) ===")
    for i in range(min(args.num_examples, len(val_df))):
        actual = " ".join(f"{targets[i, j]:6.0f}(実)" for j in range(4))
        pred = " ".join(f"{preds[i, j]:6.0f}(予)" for j in range(4))
        print(f"[{i:2d}] {actual}")
        print(f"     {pred}")


if __name__ == "__main__":
    main()
