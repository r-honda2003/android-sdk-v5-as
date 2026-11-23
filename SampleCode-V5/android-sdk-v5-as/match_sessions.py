"""
match_sessions.py

進捗率（累積移動距離を総距離で割ったもの）を用いて、
複数のセッションのフレームを基準セッションのフレームに紐付ける。

1. 各セッション(session_id > 0)に対し、開始地点からの累積移動距離を計算。
2. それを各セッションの総距離で割り、0〜1の「進捗率(progress_rate)」に変換。
3. 基準セッション(例: session_id=1)を「地図(Plan)」とし、
   他の各セッションの全フレームに対し、進捗率が最も近い基準フレームを紐付ける。
4. 出力CSVには、各フレームの image_path に加え、対応する基準フレームの ref_image_path 列を追加。

使い方:
    python match_sessions.py --telemetry telemetry_with_sessions.csv --ref_session_id 1 --output matched_telemetry.csv
"""

import argparse
import math
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Match session frames based on progress rate.")
    parser.add_argument("--telemetry", required=True, help="Path to telemetry_with_sessions.csv")
    parser.add_argument("--ref_session_id", type=int, default=1, help="基準とするセッションID(地図として使うもの)")
    parser.add_argument("--output", default="matched_telemetry.csv", help="出力CSVパス")
    parser.add_argument("--exclude_sessions", type=int, nargs="+", default=[], help="除外する session_id (例: --exclude_sessions 3)")
    parser.add_argument("--use_sessions", type=int, nargs="+", default=[], help="使用する session_id (例: --use_sessions 1 2 4 5)")
    return parser.parse_args()

def calculate_progress(df_session):
    """1つのセッションのデータフレームに対し、各行の進捗率(0-1)を計算して返す。"""
    df_session = df_session.sort_values("time").copy()
    times = df_session["time"].to_numpy()
    vx = df_session["vx"].to_numpy()
    vy = df_session["vy"].to_numpy()

    cum_dist = [0.0]
    for i in range(1, len(df_session)):
        dt = (times[i] - times[i-1]) / 1000.0
        if dt < 0: dt = 0.0
        dn = vx[i-1] * dt
        de = vy[i-1] * dt
        dist = math.hypot(dn, de)
        cum_dist.append(cum_dist[-1] + dist)

    total_dist = cum_dist[-1]
    if total_dist < 0.001:
        progress = [0.0] * len(df_session)
    else:
        progress = [d / total_dist for d in cum_dist]

    df_session["cum_dist_m"] = cum_dist
    df_session["progress_rate"] = progress
    return df_session, total_dist

def main():
    args = parse_args()
    df = pd.read_csv(args.telemetry)

    # 有効な行のみ抽出
    df = df[df["image_path"].notna() & (df["image_path"].astype(str).str.strip() != "")].copy()
    df = df[df["session_id"] != 0].copy()

    if args.exclude_sessions:
        print(f"除外対象のsession_id: {args.exclude_sessions}")
        df = df[~df["session_id"].isin(args.exclude_sessions)].copy()
    elif args.use_sessions:
        print(f"指定された使用対象のsession_id: {args.use_sessions}")
        df = df[df["session_id"].isin(args.use_sessions)].copy()

    sessions = df["session_id"].unique()
    if args.ref_session_id not in sessions:
        raise ValueError(f"基準セッションID {args.ref_session_id} がデータ内に存在しません。")

    print(f"セッション数: {len(sessions)}")

    # 1. 全セッションの進捗率を計算
    processed_sessions = {}
    for sid in sessions:
        session_df, total_dist = calculate_progress(df[df["session_id"] == sid])
        processed_sessions[sid] = session_df
        print(f"Session {sid}: 総距離 {total_dist:.2f} m, 行数 {len(session_df)}")

    # 基準セッションの準備
    ref_df = processed_sessions[args.ref_session_id]
    ref_progress = ref_df["progress_rate"].to_numpy()
    ref_images = ref_df["image_path"].to_numpy()

    # 2. 他のセッションの各フレームに対し、基準セッションで進捗率が最も近いフレームを検索
    all_matched_rows = []
    for sid in sessions:
        curr_df = processed_sessions[sid]

        # 最近傍探索 (進捗率ベース)
        # np.searchsortedを使って効率的に検索
        indices = np.searchsorted(ref_progress, curr_df["progress_rate"].to_numpy())

        # 範囲外のクリップと、左右どちらがより近いかの微調整
        indices = np.clip(indices, 1, len(ref_progress) - 1)
        left_dist = np.abs(curr_df["progress_rate"].to_numpy() - ref_progress[indices - 1])
        right_dist = np.abs(curr_df["progress_rate"].to_numpy() - ref_progress[indices])

        # より近い方のインデックスを選択
        best_indices = np.where(left_dist < right_dist, indices - 1, indices)

        curr_df["ref_image_path"] = ref_images[best_indices]
        # 基準セッション自体の場合は、ref_image_path は自分自身になる

        all_matched_rows.append(curr_df)

    # 全セッションを統合して保存
    final_df = pd.concat(all_matched_rows).sort_values(["session_id", "time"])
    final_df.to_csv(args.output, index=False)

    print(f"\nマッチング完了。出力: {args.output}")
    print(f"基準セッション: {args.ref_session_id}")

if __name__ == "__main__":
    main()
