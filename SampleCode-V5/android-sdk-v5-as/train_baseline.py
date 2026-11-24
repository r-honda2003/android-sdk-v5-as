"""
train_baseline.py

Teach & Repeat のベースラインモデル(マップ照合なし)を学習する。
「今のカメラ画像」+「直近のスティック入力履歴」から「スティック4軸」を予測する模倣学習モデル。

設計:
- telemetry CSV(image_path列とsession_id列を含むもの。split_sessions.pyの出力を想定)のうち、
  image_pathが有効かつsession_id != 0(着陸中/ノイズでない)行を学習データとして使う
- train/valは「セッション単位」で分割する。1つのセッション丸ごとをvalとして取っておく
- モデルは事前学習済みMobileNetV2(特徴抽出部分は凍結)+ スティック履歴を連結した回帰ヘッド
- NEW: 「今の画像だけでは、教示時にたまたま操作していたのか、何もしていなかったのか
  区別がつかない」問題に対応するため、直近history_lenステップ分のスティック入力履歴を
  追加の入力として与える。履歴はセッションの境目をまたいでは参照しない(session_idごとに独立)。
- スティック値(-660〜660)は-1〜1に正規化して学習する。

使い方:
    python train_baseline.py --telemetry telemetry_with_sessions.csv --images_dir images_20260827_184612 \
        --output_model baseline_model.pt --epochs 20 --val_session_id 5 --history_len 5
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
    parser = argparse.ArgumentParser(description="Train a baseline image+history-to-stick behavior cloning model.")
    parser.add_argument("--telemetry", required=True, help="Path to telemetry_with_sessions.csv (image_path, session_id列を含むもの)")
    parser.add_argument("--images_dir", required=True, help="Path to images_<sessionId>/ directory")
    parser.add_argument("--output_model", default="baseline_model.pt", help="Output model checkpoint path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--val_session_id", type=int, default=-1,
        help="この session_id を丸ごとvalidationにする。-1(デフォルト)なら最大のsession_idを自動選択",
    )
    parser.add_argument(
        "--history_len", type=int, default=5,
        help="直近何ステップ分のスティック入力履歴をモデルに与えるか。デフォルト5(約1秒分、200ms間隔想定)",
    )
    parser.add_argument(
        "--exclude_sessions", type=int, nargs="+", default=[],
        help="学習・検証から除外する session_id (例: --exclude_sessions 3)",
    )
    parser.add_argument(
        "--use_sessions", type=int, nargs="+", default=[],
        help="学習・検証に使用する session_id のリスト (例: --use_sessions 1 2 4 5)",
    )
    parser.add_argument(
        "--only_interventions", action="store_true", default=True,
        help="HG-DAgGER/Selective Experience Replay: 2回目以降のAI飛行セッションで人間が介入した瞬間(+前後ウィンドウ)のみを抽出して学習に利用する (デフォルト: True)",
    )
    parser.add_argument(
        "--no_only_interventions", action="store_false", dest="only_interventions",
        help="人間介入フィルタリングを無効化し、AI飛行セッションの全フレームを学習に利用する",
    )
    parser.add_argument(
        "--window_before", type=int, default=5,
        help="人間介入発生直前の何ステップ(フレーム)を学習データに含めるか (デフォルト: 5 = 約1秒前)",
    )
    parser.add_argument(
        "--window_after", type=int, default=5,
        help="人間介入終了直後の何ステップ(フレーム)を学習データに含めるか (デフォルト: 5 = 約1秒後)",
    )
    parser.add_argument(
        "--base_session_id", type=int, default=1,
        help="全フレームを無条件保持する初期手動教示セッションのID (デフォルト: 1)",
    )
    return parser.parse_args()


def add_stick_history_columns(df: pd.DataFrame, history_len: int) -> pd.DataFrame:
    """各行に、同じsession_id内での直近history_lenステップ分のスティック値を列として追加する。
    セッション先頭付近で履歴が足りない場合は0で埋める(=「直前は何もしていなかった」扱い)。
    """
    df = df.sort_values(["session_id", "time"]).reset_index(drop=True)
    for col in STICK_COLS:
        for lag in range(1, history_len + 1):
            df[f"{col}_lag{lag}"] = df.groupby("session_id")[col].shift(lag).fillna(0)
    return df


def history_columns(history_len: int):
    return [f"{col}_lag{lag}" for lag in range(1, history_len + 1) for col in STICK_COLS]


class ImageStickDataset(Dataset):
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
        img_path = os.path.join(self.images_dir, row["image_path"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        history = torch.tensor(
            [row[c] / STICK_MAX for c in self.hist_cols],
            dtype=torch.float32,
        )

        target = torch.tensor(
            [row[c] / STICK_MAX for c in STICK_COLS],
            dtype=torch.float32,
        )
        return image, history, target


class BaselinePolicy(nn.Module):
    """MobileNetV2バックボーン(事前学習・特徴抽出部分は凍結) + スティック履歴 + 回帰ヘッド"""

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
        self.head = nn.Sequential(
            nn.Linear(1280 + history_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 4),
            nn.Tanh(),  # 出力を[-1, 1]に収める(正規化したスティック値の範囲と一致)
        )

    def forward(self, x, history):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.cat([x, history], dim=1)
        x = self.head(x)
        return x


def filter_human_interventions(
    df: pd.DataFrame,
    window_before: int = 5,
    window_after: int = 5,
    base_session_id: int = 1
) -> pd.DataFrame:
    """
    HG-DAgGER / Selective Experience Replay:
    - 初期手動教示セッション (session_id == base_session_id) は全データ保持
    - 以降のAI飛行セッション (session_id != base_session_id) は、
      human_override_active == 1 (またはプロポ手動入力あり) のフレームおよびその前後 (window_before, window_after) のみを抽出
    """
    if df.empty:
        return df

    filtered_dfs = []
    print("\n--- [HG-DAgGER / Selective Experience Replay フィルタリング開始] ---")

    for session_id, group in df.groupby("session_id", sort=True):
        if session_id == base_session_id:
            print(f"  Session {session_id} (初期教示データ): 全 {len(group)} 行を保持")
            filtered_dfs.append(group)
        else:
            # 1) human_override_active == 1 フラグチェック
            is_override = pd.Series(False, index=group.index)
            if "human_override_active" in group.columns:
                is_override = is_override | (group["human_override_active"] == 1)

            # 2) プロポ手動スティック入力の閾値チェック (|stick| > 50)
            for col in STICK_COLS:
                if col in group.columns:
                    is_override = is_override | (group[col].abs() > 50)

            override_count = is_override.sum()
            if override_count == 0:
                print(f"  Session {session_id} (AI飛行): 人間介入なしのためスキップ (0/{len(group)} 行)")
                continue

            # 介入フレームとその前後 (window_before, window_after) のインデックスを収集
            group_indices = group.index.tolist()
            idx_to_pos = {idx: pos for pos, idx in enumerate(group_indices)}
            selected_indices = set()

            for override_idx in group.index[is_override]:
                pos = idx_to_pos[override_idx]
                start_pos = max(0, pos - window_before)
                end_pos = min(len(group_indices) - 1, pos + window_after)
                for p in range(start_pos, end_pos + 1):
                    selected_indices.add(group_indices[p])

            selected_df = group.loc[sorted(list(selected_indices))].copy()
            print(
                f"  Session {session_id} (AI飛行+介入): 全 {len(group)} 行中 "
                f"{len(selected_df)} 行を抽出 (直接介入: {override_count} 行 + 前後ウィンドウ)"
            )
            filtered_dfs.append(selected_df)

    if not filtered_dfs:
        print("[Warning] フィルタリング後のデータが0件のため、全データを保持します。")
        return df

    res_df = pd.concat(filtered_dfs).sort_values(["session_id", "time"]).reset_index(drop=True)
    print(f"HG-DAgGER 抽出完了: 全体 {len(df)} 行 -> {len(res_df)} 行へ縮小")
    print("-----------------------------------------------------------\n")
    return res_df


def load_dataset_df(telemetry_path: str, history_len: int) -> pd.DataFrame:
    df = pd.read_csv(telemetry_path)
    if "session_id" not in df.columns:
        raise ValueError(
            "session_id列が見つかりません。先に split_sessions.py で telemetry_with_sessions.csv を作成してください。"
        )
    df = df[df["image_path"].notna() & (df["image_path"].astype(str).str.strip() != "")].copy()
    df = df[df["session_id"] != 0].copy()  # 着陸中/ノイズ区間を除外
    df = add_stick_history_columns(df, history_len)
    return df


def main():
    args = parse_args()
    df = load_dataset_df(args.telemetry, args.history_len)

    if args.only_interventions:
        df = filter_human_interventions(
            df,
            window_before=args.window_before,
            window_after=args.window_after,
            base_session_id=args.base_session_id
        )

    if args.exclude_sessions:
        print(f"除外対象のsession_id: {args.exclude_sessions}")
        df = df[~df["session_id"].isin(args.exclude_sessions)].copy()
    elif args.use_sessions:
        print(f"指定された使用対象のsession_id: {args.use_sessions}")
        df = df[df["session_id"].isin(args.use_sessions)].copy()

    print(f"有効サンプル数: {len(df)}")

    available_sessions = sorted(df["session_id"].unique())
    print(f"学習・検証に使用するsession_id: {available_sessions}")

    val_session_id = args.val_session_id
    if len(available_sessions) == 1 or val_session_id == 0:
        train_df = df.copy().reset_index(drop=True)
        val_df = df.copy().reset_index(drop=True)
        print(f"単一セッションデータのため全 {len(train_df)} 枚を学習・検証に使用します")
    else:
        if val_session_id == -1:
            val_session_id = max(available_sessions)
        if val_session_id not in available_sessions:
            raise ValueError(f"session_id {val_session_id} はデータ内に存在しません。available: {available_sessions}")

        train_df = df[df["session_id"] != val_session_id].reset_index(drop=True)
        val_df = df[df["session_id"] == val_session_id].reset_index(drop=True)
        print(f"train sessions: {[s for s in available_sessions if s != val_session_id]} ({len(train_df)}枚)")
        print(f"val session: {val_session_id} ({len(val_df)}枚)")
    print(f"history_len: {args.history_len} (=履歴 {len(STICK_COLS)*args.history_len}次元を追加入力)")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ImageStickDataset(train_df, args.images_dir, transform, args.history_len)
    val_ds = ImageStickDataset(val_df, args.images_dir, transform, args.history_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = BaselinePolicy(history_len=args.history_len, freeze_backbone=True).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, history, targets in train_loader:
            images, history, targets = images.to(device), history.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(images, history)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, history, targets in val_loader:
                images, history, targets = images.to(device), history.to(device), targets.to(device)
                preds = model(images, history)
                loss = criterion(preds, targets)
                val_loss += loss.item() * images.size(0)
        val_loss /= max(1, len(val_ds))

        print(f"epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

    torch.save({"state_dict": model.state_dict(), "history_len": args.history_len}, args.output_model)
    print(f"\nモデルを保存しました: {args.output_model}")


if __name__ == "__main__":
    main()
