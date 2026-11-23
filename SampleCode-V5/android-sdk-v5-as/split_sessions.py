"""
split_sessions.py

離着陸を繰り返しながら連続記録した1本のtelemetry CSVから、
「スティック操作が全軸ほぼゼロ(中立)状態が一定回数(デフォルト50行以上)連続している区間」を検出し、
それ以外の飛行区間ごとに session_id を振り分ける。

考え方:
    stick_left_v, stick_left_h, stick_right_v, stick_right_h の絶対値がすべて10未満の状態を「中立候補」とし、
    これが min_neutral_count(デフォルト50行) 以上連続した場合に「待機/着陸区間 (session_id = 0)」とみなす。
    それ以外の有効な連続飛行区間に 1 から順に session_id を割り振る。

使い方:
    python split_sessions.py --telemetry telemetry_20261118_112806.csv --output telemetry_with_sessions.csv
"""

import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Split telemetry log into flight sessions based on stick neutral duration.")
    parser.add_argument("--telemetry", required=True, help="Path to the merged telemetry CSV")
    parser.add_argument("--output", default="telemetry_with_sessions.csv", help="Output CSV path")
    parser.add_argument("--stick_threshold", type=int, default=10, help="この値未満を入力ゼロ(中立)とみなす閾値。デフォルト10")
    parser.add_argument("--min_neutral_count", type=int, default=50, help="スティック中立がこの回数(行)以上連続したら除外区間(session_id=0)と判定。デフォルト50行")
    parser.add_argument("--min_session_duration_s", type=float, default=3.0, help="これより短い飛行区間はノイズとして除外(session_id=0)。デフォルト3.0秒")
    return parser.parse_args()


def detect_sessions_by_stick(df: pd.DataFrame, stick_th: int, min_neutral_count: int, min_session_s: float):
    df = df.sort_values("time").reset_index(drop=True)
    times = df["time"].to_numpy()

    sl_v = df["stick_left_v"].abs().to_numpy()
    sl_h = df["stick_left_h"].abs().to_numpy()
    sr_v = df["stick_right_v"].abs().to_numpy()
    sr_h = df["stick_right_h"].abs().to_numpy()

    # 全軸の絶対値が stick_th 未満か判断
    is_neutral_candidate = (sl_v < stick_th) & (sl_h < stick_th) & (sr_v < stick_th) & (sr_h < stick_th)

    # min_neutral_count (例:50行) 以上連続して中立なら除外区間(is_landed = True)
    is_landed = [False] * len(df)
    i = 0
    while i < len(df):
        if is_neutral_candidate[i]:
            j = i
            while j < len(df) and is_neutral_candidate[j]:
                j += 1
            count = j - i
            if count >= min_neutral_count:
                for k in range(i, j):
                    is_landed[k] = True
            i = j
        else:
            i += 1

    # 飛行区間に session_id を割り振る
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

    if "session_id" in df.columns and len(df["session_id"].unique()) > 1:
        print(f"入力CSVに明示的な session_id ({sorted(df['session_id'].unique())}) が定義されているため、再分割をスキップします。")
        df.to_csv(args.output, index=False)
        return

    df, sessions_info = detect_sessions_by_stick(
        df, args.stick_threshold, args.min_neutral_count, args.min_session_duration_s
    )

    print(f"\n検出されたセッション数: {len(sessions_info)} (中立連続閾値: {args.min_neutral_count}行)")
    print(f"{'session_id':>10} {'行数':>6} {'秒数':>8}")
    for sid, start, end, duration, n_rows in sessions_info:
        print(f"{sid:>10} {n_rows:>6} {duration:>8.1f}")

    n_landed = (df["session_id"] == 0).sum()
    print(f"\n中立待機/ノイズとして除外される行数: {n_landed} / {len(df)}")

    df.to_csv(args.output, index=False)
    print(f"\n書き出しました: {args.output}")


if __name__ == "__main__":
    main()
