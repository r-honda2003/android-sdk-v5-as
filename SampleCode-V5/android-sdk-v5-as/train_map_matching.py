"""
train_map_matching.py

マップ照合（基準フレームとのペア）を用いた Teach & Repeat モデルを学習する。
「今のカメラ画像」+「進捗が対応する基準フレーム(Map)の画像」+「スティック履歴」から「スティック4軸」を予測する。

設計:
- match_sessions.py で作成した matched_telemetry.csv を使用する (ref_image_path列を含む)。
- モデルは今の画像と基準画像の2つを入力とし、それぞれの特徴量を MobileNetV2 で抽出して結合する。
- スティック履歴も同様に結合して回帰ヘッドに渡す。

使い方:
    python train_map_matching.py --telemetry matched_telemetry.csv --images_dir images_20260827_184612 \
        --output_model map_matching_model.pt --epochs 20 --val_session_id 5 --history_len 5
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

STICK_MAX = 660.0
STICK_COLS = ["stick_left_v", "stick_left_h", "stick_right_v", "stick_right_h"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a map-matching image+history-to-stick model.")
    parser.add_argument("--telemetry", required=True, help="Path to matched_telemetry.csv (ref_image_path列を含むもの)")
    parser.add_argument("--images_dir", required=True, help="Path to images_<sessionId>/ directory")
    parser.add_argument("--output_model", default="map_matching_model.pt", help="Output model checkpoint path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_session_id", type=int, default=-1)
    parser.add_argument("--history_len", type=int, default=5)
    return parser.parse_args()


def add_stick_history_columns(df: pd.DataFrame, history_len: int) -> pd.DataFrame:
    df = df.sort_values(["session_id", "time"]).reset_index(drop=True)
    for col in STICK_COLS:
        for lag in range(1, history_len + 1):
            df[f"{col}_lag{lag}"] = df.groupby("session_id")[col].shift(lag).fillna(0)
    return df


def history_columns(history_len: int):
    return [f"{col}_lag{lag}" for lag in range(1, history_len + 1) for col in STICK_COLS]


class MapMatchingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: str, transform, history_len: int):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.history_len = history_len
        self.hist_cols = history_columns(history_len)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 今の画像
        img_path = os.path.join(self.images_dir, row["image_path"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # 基準フレーム(Map)の画像
        ref_path = os.path.join(self.images_dir, row["ref_image_path"])
        ref_image = Image.open(ref_path).convert("RGB")
        ref_image = self.transform(ref_image)

        history = torch.tensor(
            [row[c] / STICK_MAX for c in self.hist_cols],
            dtype=torch.float32,
        )

        target = torch.tensor(
            [row[c] / STICK_MAX for c in STICK_COLS],
            dtype=torch.float32,
        )
        return image, ref_image, history, target


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

        history_dim = history_len * len(STICK_COLS)
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


def load_dataset_df(telemetry_path: str, history_len: int) -> pd.DataFrame:
    df = pd.read_csv(telemetry_path)
    if "ref_image_path" not in df.columns:
        raise ValueError("ref_image_path列が見つかりません。先に match_sessions.py を実行してください。")
    df = add_stick_history_columns(df, history_len)
    return df


def main():
    args = parse_args()
    df = load_dataset_df(args.telemetry, args.history_len)
    print(f"有効サンプル数: {len(df)}")

    available_sessions = sorted(df["session_id"].unique())
    val_session_id = args.val_session_id
    if val_session_id == -1:
        val_session_id = max(available_sessions)

    train_df = df[df["session_id"] != val_session_id].reset_index(drop=True)
    val_df = df[df["session_id"] == val_session_id].reset_index(drop=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = MapMatchingDataset(train_df, args.images_dir, transform, args.history_len)
    val_ds = MapMatchingDataset(val_df, args.images_dir, transform, args.history_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MapMatchingPolicy(history_len=args.history_len, freeze_backbone=True).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, ref_images, history, targets in train_loader:
            images, ref_images, history, targets = images.to(device), ref_images.to(device), history.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(images, ref_images, history)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, ref_images, history, targets in val_loader:
                images, ref_images, history, targets = images.to(device), ref_images.to(device), history.to(device), targets.to(device)
                preds = model(images, ref_images, history)
                loss = criterion(preds, targets)
                val_loss += loss.item() * images.size(0)
        val_loss /= max(1, len(val_ds))

        print(f"epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

    torch.save({"state_dict": model.state_dict(), "history_len": args.history_len}, args.output_model)
    print(f"\nモデルを保存しました: {args.output_model}")


if __name__ == "__main__":
    main()
