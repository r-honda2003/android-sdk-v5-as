"""
split_sessions.py

離着陸を繰り返しながら連続記録した1本のtelemetry CSVから、
「高度・速度がほぼゼロになっている区間(=着陸中)」を検出して、
飛行区間ごとに session_id を振り分ける。

考え方:
    alt(高度)とspeed(水平速度)が両方とも閾値未満の状態が一定時間(min_landed_duration_s)
    以上続いたら「着陸中」とみなし、それを周と周の境目とする。
    着陸中でない連続区間(=1回の飛行)に、1から順にsession_idを割り振る。
    着陸中の行は session_id = 0 とし、学習時には除外する想定。

使い方:
    python split_sessions.py --telemetry telemetry_merged.csv --output telemetry_with_sessions.csv

出力:
    元のCSVに session_id 列を追加したもの。
    加えて、各セッションの行数・推定飛行時間・推定距離をコンソールに一覧表示する。
"""

import argparse
import math

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Split a single merged telemetry log into multiple flight sessions.")
    parser.add_argument("--telemetry", required=True, help="Path to the merged telemetry CSV")
    parser.add_argument("--output", default="telemetry_with_sessions.csv", help="Output CSV path")
    parser.add_argument("--alt_threshold_m", type=float, default=0.15, help="この高度(m)未満を'着陸中'候補とみなす。デフォルト0.15")
    parser.add_argument("--speed_threshold_mps", type=float, default=0.05, help="この速度(m/s)未満を'静止中'候補とみなす。デフォルト0.05")
    parser.add_argument("--min_landed_duration_s", type=float, default=2.0, help="着陸中とみなす最小継続時間(秒)。デフォルト2.0")
    parser.add_argument("--min_session_duration_s", type=float, default=10.0, help="これより短い飛行区間はノイズとして除外(session_id=0)。デフォルト10.0")
    return parser.parse_args()


def detect_sessions(df: pd.DataFrame, alt_th: float, speed_th: float,
                     min_landed_s: float, min_session_s: float) -> pd.DataFrame:
    df = df.sort_values("time").reset_index(drop=True)
    times = df["time"].to_numpy()
    alt = df["alt"].to_numpy()
    vx = df["vx"].to_numpy()
    vy = df["vy"].to_numpy()
    speed = [math.hypot(vx[i], vy[i]) for i in range(len(df))]

    is_landed_candidate = [(alt[i] < alt_th) and (speed[i] < speed_th) for i in range(len(df))]

    # 連続する"着陸候補"区間のうち、min_landed_duration_s以上続くものだけを本物の"着陸中"とする
    is_landed = [False] * len(df)
    i = 0
    while i < len(df):
        if is_landed_candidate[i]:
            j = i
            while j < len(df) and is_landed_candidate[j]:
                j += 1
            duration = (times[j - 1] - times[i]) / 1000.0
            if duration >= min_landed_s:
                for k in range(i, j):
                    is_landed[k] = True
            i = j
        else:
            i += 1

    # 着陸中でない連続区間にsession_idを割り振る
    session_id = [0] * len(df)
    current_id = 0
    i = 0
    sessions_info = []
    while i < len(df):
        if not is_landed[i]:
            j = i
            while j < len(df) and not is_landed[j]:
                j += 1
            duration = (times[j - 1] - times[i]) / 1000.0
            if duration >= min_session_s:
                current_id += 1
                for k in range(i, j):
                    session_id[k] = current_id
                sessions_info.append((current_id, i, j, duration, j - i))
            i = j
        else:
            i += 1

    df["session_id"] = session_id
    return df, sessions_info


def main():
    args = parse_args()
    df = pd.read_csv(args.telemetry)
    print(f"telemetry総行数: {len(df)}")

    df, sessions_info = detect_sessions(
        df, args.alt_threshold_m, args.speed_threshold_mps,
        args.min_landed_duration_s, args.min_session_duration_s,
    )

    print(f"\n検出されたセッション数: {len(sessions_info)}")
    print(f"{'session_id':>10} {'行数':>6} {'秒数':>8}")
    for sid, start, end, duration, n_rows in sessions_info:
        print(f"{sid:>10} {n_rows:>6} {duration:>8.1f}")

    n_landed = (df["session_id"] == 0).sum()
    print(f"\n着陸中/ノイズとして除外される行数: {n_landed} / {len(df)}")

    df.to_csv(args.output, index=False)
    print(f"\n書き出しました: {args.output}")


if __name__ == "__main__":
    main()
