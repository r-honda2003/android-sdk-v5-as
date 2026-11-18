"""
compute_calculated_sticks.py

既存の telemetry CSV ファイル（A~Q列）から、C~H列（vx, vy, vz, pitch, roll, yaw, time）を使って
逆算したスティック制御量（-660 ~ +660）を計算し、R~U列として追加・保存します。

R列: calc_stick_left_v  (スロットル / 上昇・下降)
S列: calc_stick_left_h  (ヨー / 左右旋回)
T列: calc_stick_right_v (ピッチ / 前進・後進)
U列: calc_stick_right_h (ロール / 左右移動)

使い方:
    python3 compute_calculated_sticks.py --input telemetry.csv --output telemetry_with_calc.csv
"""

import argparse
import math
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate -660~+660 stick controls from telemetry (C~H columns) and append to R~U columns.")
    parser.add_argument("--input", required=True, help="Input telemetry CSV path")
    parser.add_argument("--output", default="", help="Output CSV path (default: overwrite or input_with_calc.csv)")
    parser.add_argument("--max_vz", type=float, default=3.0, help="Max vertical speed (m/s), default 3.0")
    parser.add_argument("--max_vxy", type=float, default=5.0, help="Max horizontal speed (m/s), default 5.0")
    parser.add_argument("--max_yaw_rate", type=float, default=25.6, help="Max yaw rate (deg/s), default 25.6")
    return parser.parse_args()


def calculate_stick_columns(df: pd.DataFrame, max_vz=3.0, max_vxy=5.0, max_yaw_rate=25.6) -> pd.DataFrame:
    df = df.copy()

    # 1. R列: calc_stick_left_v (スロットル: 上昇速度 -vz より算出し、-660 ~ 660 にクリップ)
    if 'vz' in df.columns:
        df['calc_stick_left_v'] = np.clip((-df['vz'] / max_vz) * 660.0, -660, 660).round().astype(int)
    else:
        df['calc_stick_left_v'] = 0

    # 2. S列: calc_stick_left_h (ヨー: yawの時間変化率 deg/s より算出)
    if 'yaw' in df.columns and 'time' in df.columns:
        dt = df['time'].diff() / 1000.0  # ms -> s
        dyaw = df['yaw'].diff()

        # -180 ~ 180 度の跨ぎ補正
        dyaw = (dyaw + 180.0) % 360.0 - 180.0
        yaw_rate = dyaw / dt
        yaw_rate = yaw_rate.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        df['calc_stick_left_h'] = np.clip((yaw_rate / max_yaw_rate) * 660.0, -660, 660).round().astype(int)
    else:
        df['calc_stick_left_h'] = 0

    # 3. T列: calc_stick_right_v (ピッチ: 前後速度 vx より算出)
    if 'vx' in df.columns:
        df['calc_stick_right_v'] = np.clip((df['vx'] / max_vxy) * 660.0, -660, 660).round().astype(int)
    else:
        df['calc_stick_right_v'] = 0

    # 4. U列: calc_stick_right_h (ロール: 左右速度 vy より算出)
    if 'vy' in df.columns:
        df['calc_stick_right_h'] = np.clip((df['vy'] / max_vxy) * 660.0, -660, 660).round().astype(int)
    else:
        df['calc_stick_right_h'] = 0

    return df


def main():
    args = parse_args()
    output_path = args.output if args.output else args.input.replace(".csv", "_with_calc.csv")

    df = pd.read_csv(args.input)
    print(f"入力CSV読み込み完了: '{args.input}' ({len(df)} 行)")

    df_calc = calculate_stick_columns(df, args.max_vz, args.max_vxy, args.max_yaw_rate)

    df_calc.to_csv(output_path, index=False)
    print(f"R~U列 (calc_stick_left_v, calc_stick_left_h, calc_stick_right_v, calc_stick_right_h) を追加しました！")
    print(f"出力ファイル: '{output_path}'")


if __name__ == "__main__":
    main()
