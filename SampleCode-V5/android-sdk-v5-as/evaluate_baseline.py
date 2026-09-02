"""
evaluate_baseline.py

学習済みのbaseline_model.ptを使い、validationセッションに対する予測誤差を、
正規化された値(-1〜1)ではなく、実際のスティック値の単位(-660〜660)で確認する。
あわせて、いくつかのサンプルで「実際の値 vs 予測値」を具体的に見せる。

使い方:
    python evaluate_baseline.py --telemetry telemetry_with_sessions.csv --images_dir images_20260827_184612 \
        --model baseline_model.pt --val_session_id 5
"""

import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from train_baseline import BaselinePolicy, ImageStickDataset, load_dataset_df, STICK_MAX, STICK_COLS


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained baseline model on a held-out session.")
    parser.add_argument("--telemetry", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--model", required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--val_session_id", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_examples", type=int, default=8, help="実際vs予測を表示するサンプル数")
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
    val_ds = ImageStickDataset(val_df, args.images_dir, transform, history_len)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselinePolicy(history_len=history_len, freeze_backbone=True).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, history, targets in val_loader:
            images, history = images.to(device), history.to(device)
            preds = model(images, history).cpu()
            all_preds.append(preds)
            all_targets.append(targets)

    preds = torch.cat(all_preds, dim=0) * STICK_MAX      # -1..1 -> -660..660
    targets = torch.cat(all_targets, dim=0) * STICK_MAX

    abs_err = (preds - targets).abs()
    mae_per_axis = abs_err.mean(dim=0)
    rmse_per_axis = ((preds - targets) ** 2).mean(dim=0).sqrt()

    print("\n=== スティック値の単位(-660〜660)での誤差 ===")
    for i, col in enumerate(STICK_COLS):
        print(f"{col:>16}: MAE={mae_per_axis[i]:7.1f}  RMSE={rmse_per_axis[i]:7.1f}")
    print(f"{'全軸平均':>16}: MAE={mae_per_axis.mean():7.1f}  RMSE={rmse_per_axis.mean():7.1f}")
    print("(参考: スティックの可動範囲は±660。誤差がこの値の何%かで感覚をつかむとよい)")

    print(f"\n=== サンプル比較(先頭{args.num_examples}件) ===")
    for i in range(min(args.num_examples, len(val_df))):
        actual = " ".join(f"{targets[i, j]:7.1f}(実)" for j in range(4))
        pred = " ".join(f"{preds[i, j]:7.1f}(予)" for j in range(4))
        print(f"{i:>4} {actual}")
        print(f"{'':>4} {pred}")


if __name__ == "__main__":
    main()
