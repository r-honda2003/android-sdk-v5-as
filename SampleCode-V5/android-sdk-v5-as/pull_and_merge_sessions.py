"""
pull_and_merge_sessions.py

接続されたAndroid端末(またはローカルディレクトリ)から
telemetry_run1_session_*.csv および images_run1_session_* フォルダを自動取得・統合し、
1つの telemetry_with_sessions.csv と images_run1 ディレクトリにまとめます。

使い方の例:
    python pull_and_merge_sessions.py [--run_prefix run1] [--output_csv telemetry_with_sessions.csv] [--output_img_dir images_run1]
"""

import argparse
import glob
import os
import re
import shutil
import subprocess

import pandas as pd

NUMERIC_COLS = [
    "time", "vx", "vy", "vz", "pitch", "roll", "yaw", "lat", "lng", "alt",
    "battery_percent", "stick_left_v", "stick_left_h", "stick_right_v", "stick_right_h",
    "calc_stick_left_v", "calc_stick_left_h", "calc_stick_right_v", "calc_stick_right_h",
    "send_stick_left_v", "send_stick_left_h", "send_stick_right_v", "send_stick_right_h",
    "human_override_active"
]


def extract_session_num(filepath: str) -> int:
    m = re.search(r"session_(\d+)", filepath)
    if m:
        return int(m.group(1))
    return 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pull and merge multi-session telemetry CSVs and image directories from device."
    )
    parser.add_argument(
        "--run_prefix",
        default="run1",
        help="対象とするセッションのプレフィックス (デフォルト: run1)",
    )
    parser.add_argument(
        "--output_csv",
        default="telemetry_with_sessions.csv",
        help="マージ後のCSV保存パス (デフォルト: telemetry_with_sessions.csv)",
    )
    parser.add_argument(
        "--output_img_dir",
        default="images_run1",
        help="画像統合先ディレクトリ名 (デフォルト: images_run1)",
    )
    parser.add_argument(
        "--skip_adb_pull",
        action="store_true",
        help="adb pull による端末からのダウンロードをスキップし、ローカルのCSVのみマージする",
    )
    return parser.parse_args()


def run_cmd(cmd: str):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return res.returncode, res.stdout.strip(), res.stderr.strip()


def pull_from_device(run_prefix: str, output_img_dir: str):
    device_path = "/sdcard/Android/data/com.dji.sampleV5.aircraft/files"
    print(f"=== [ADB] 端末からのデータ取得を開始: {device_path} ===")

    code, out, _ = run_cmd(f"adb shell 'ls {device_path}/telemetry_{run_prefix}_session_*.csv {device_path}/telemetry_20261123_*.csv 2>/dev/null'")
    if code == 0 and out:
        remote_csvs = [line.strip() for line in out.splitlines() if line.strip()]
        for r_csv in remote_csvs:
            print(f"  Pulling CSV: {r_csv}")
            run_cmd(f"adb pull '{r_csv}' .")
    else:
        print("  端末上に最新のセッションファイルが見つかりませんでした。")

    code, out, _ = run_cmd(f"adb shell 'ls -d {device_path}/images_{run_prefix}_session_* {device_path}/images_20261123_* 2>/dev/null'")
    if code == 0 and out:
        remote_img_dirs = [line.strip() for line in out.splitlines() if line.strip()]
        for r_dir in remote_img_dirs:
            print(f"  Pulling Image Directory: {r_dir}")
            run_cmd(f"adb pull '{r_dir}' .")


def merge_sessions(run_prefix: str, output_csv: str, output_img_dir: str):
    os.makedirs(output_img_dir, exist_ok=True)

    csv_pattern = f"telemetry_{run_prefix}_session_*.csv"
    local_csvs = sorted(glob.glob(csv_pattern), key=extract_session_num)

    has_session_1 = any(extract_session_num(f) == 1 for f in local_csvs)
    if not has_session_1:
        ts_csvs = sorted(glob.glob("telemetry_20261123_103615.csv")) or sorted(glob.glob("telemetry_20*.csv"))
        if ts_csvs:
            local_csvs.insert(0, ts_csvs[0])

    # 介入飛行（Repeat時の手動割り込み）などの追加タイムスタンプ付きCSVも含める (今日のデータのみ)
    extra_csvs = [f for f in sorted(glob.glob("telemetry_20261123_*.csv")) if f not in local_csvs and f not in ["telemetry_with_sessions.csv", "matched_telemetry.csv"]]
    local_csvs.extend(extra_csvs)

    if not local_csvs:
        raise FileNotFoundError(f"マージ対象のテレメトリCSVが見つかりません。")

    print(f"\n=== [Merge] セッションデータの統合処理を開始 ({len(local_csvs)} ファイル検出) ===")
    dfs = []

    for default_idx, csv_file in enumerate(local_csvs, start=1):
        if "session_" in csv_file:
            session_idx = extract_session_num(csv_file)
        else:
            session_idx = default_idx

        df = pd.read_csv(csv_file)
        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[df["time"].notna()].copy()
        df["time"] = df["time"].astype("int64")
        df["session_id"] = session_idx

        # CSV内で参照されている画像のみを収集・コピー
        for img_name in df["image_path"].dropna().unique():
            img_name_str = str(img_name).strip()
            if not img_name_str:
                continue
            dest_file = os.path.join(output_img_dir, img_name_str)
            if not os.path.exists(dest_file):
                for src_dir in sorted(glob.glob("images_*")):
                    if src_dir == output_img_dir:
                        continue
                    src_file = os.path.join(src_dir, img_name_str)
                    if os.path.exists(src_file):
                        shutil.copy(src_file, dest_file)
                        break

        dfs.append(df)
        valid_img_rows = len(df[df["image_path"].notna() & (df["image_path"].astype(str).str.strip() != "")])
        print(f"  Session {session_idx} ({os.path.basename(csv_file)}): 行数 = {len(df)}, 有効画像数 = {valid_img_rows}")

    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(output_csv, index=False)

    print(f"\n==========================================")
    print(f" 統合完了:")
    print(f"   CSV出力     : {output_csv} (総行数: {len(merged_df)})")
    print(f"   画像フォルダ : {output_img_dir} (総画像数: {len(glob.glob(os.path.join(output_img_dir, '*.jpg')))} 枚)")
    print(f"==========================================")


def main():
    args = parse_args()
    if not args.skip_adb_pull:
        pull_from_device(args.run_prefix, args.output_img_dir)
    merge_sessions(args.run_prefix, args.output_csv, args.output_img_dir)


if __name__ == "__main__":
    main()
