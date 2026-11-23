"""
merge_sessions.py

個別で記録されたセッションCSVファイル (例: Session 1, Session 2, Session 3, Session 4, Session 5) を読み込み、
それぞれの行に session_id (1, 2, 3, 4, 5...) を付与して 1 つの CSV ファイル (telemetry_with_sessions.csv) に統合する。

使い方の例:
    python merge_sessions.py --files telemetry_20261123_103615.csv telemetry_run1_session_2.csv telemetry_run1_session_3.csv telemetry_run1_session_4.csv telemetry_run1_session_5.csv --output telemetry_with_sessions.csv
"""

import argparse
import pandas as pd

NUMERIC_COLS = [
    "time", "vx", "vy", "vz", "pitch", "roll", "yaw", "lat", "lng", "alt",
    "battery_percent", "stick_left_v", "stick_left_h", "stick_right_v", "stick_right_h",
    "calc_stick_left_v", "calc_stick_left_h", "calc_stick_right_v", "calc_stick_right_h",
    "send_stick_left_v", "send_stick_left_h", "send_stick_right_v", "send_stick_right_h",
    "human_override_active"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge separate session CSV files into a single telemetry CSV with session_id column."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="Input session CSV files in order (Session 1, 2, 3, ...)",
    )
    parser.add_argument(
        "--output", default="telemetry_with_sessions.csv", help="Output merged CSV path"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dfs = []
    print(f"マージ対象ファイル数: {len(args.files)}")
    for idx, filepath in enumerate(args.files, start=1):
        df = pd.read_csv(filepath)
        # 重複ヘッダー行などを除去するために数値列を数値化
        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[df["time"].notna()].copy()
        df["time"] = df["time"].astype("int64")
        df["session_id"] = idx

        valid_count = len(
            df[df["image_path"].notna() & (df["image_path"].astype(str).str.strip() != "")]
        )
        print(f"Session {idx}: {filepath} -> 有効行数: {len(df)}, 有効画像行数: {valid_count}")
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(args.output, index=False)
    print(
        f"\n全 {len(dfs)} セッションのデータをマージして {args.output} に保存しました（総行数: {len(merged_df)}）。"
    )


if __name__ == "__main__":
    main()
