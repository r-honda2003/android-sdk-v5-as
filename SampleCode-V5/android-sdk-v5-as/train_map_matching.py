"""
train_map_matching.py

自作 2D-CNN (Drone2DCNN) + マップ照合を用いた Teach & Repeat モデルを学習する。
「今のカメラ画像」+「進捗が対応する基準フレーム(Map)の画像」+「スティック履歴」から「スティック4軸」を予測する。
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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


# =========================================================
# 1. 自作 2D-CNN 特徴抽出クラス (Drone2DCNN)
# =========================================================
class Drone2DCNN(nn.Module):
    def __init__(self, output_feature_dim=1280):
        super().__init__()

        self.features = nn.Sequential(
            # 第1ブロック: 基本的な線や色合いを検出 (224x224 -> 112x112)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 第2ブロック: 形状の検出 (112x112 -> 56x56)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 第3ブロック: 物体パーツの検出 (56x56 -> 28x28)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 第4ブロック: 高次特徴の検出 (28x28 -> 14x14)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, output_feature_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        feat = self.fc(x)
        return feat


# =========================================================
# 2. E2E 全体モデル (自作 2D-CNN 搭載)
# =========================================================
class MapMatchingPolicy(nn.Module):
    def __init__(self, history_len: int = 5):
        super().__init__()

        # バックボーンを自作 2D-CNN に設定
        self.backbone = Drone2DCNN(output_feature_dim=1280)

        history_dim = history_len * len(STICK_COLS) # 20次元

        # 1280 (curr_img) + 1280 (ref_img) + 20 (history)
        self.head = nn.Sequential(
            nn.Linear(1280 * 2 + history_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 4),  # 出力: 4軸
            nn.Tanh(),
        )

    def forward(self, x, ref_x, history):
        # 自作 2D-CNN で特徴抽出
        feat = self.backbone(x)
        ref_feat = self.backbone(ref_x)

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
    model = MapMatchingPolicy(history_len=args.history_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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
    print(f"\n自作2D-CNNモデルを保存しました: {args.output_model}")


if __name__ == "__main__":
    main()
