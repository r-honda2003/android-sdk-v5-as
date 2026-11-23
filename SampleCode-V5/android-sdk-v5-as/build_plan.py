"""
build_plan.py

Teach走行のtelemetry_<sessionId>.csv（image_path列を含む、撮影時点で同期済みのもの）から、
0.5秒間隔で間引いた「基準フレーム列（Plan）」を構築する。

各基準フレームには、教示開始地点からの相対移動量（vx, vyを積分した推定値）を付与する。
DJIのKeyAircraftVelocityはNED(北・東・下)座標系の値のため、
機体の向き(yaw)に関わらずそのまま積分すれば、ワールド座標系での相対変位が得られる。
絶対座標(GPS)は屋内で使えないため、この積分値を「進捗の目安」として使う。

NEW: 記録の最後に「速度がほぼゼロの状態」が一定時間続く場合(=着陸後に録画を止め忘れた等)、
その区間をPlanから打ち切る。教示走行の途中で一時的にホバリングする区間は打ち切り対象にしない
(末尾から連続でゼロ速度が続く区間のみが対象)。

使い方:
    python build_plan.py --telemetry telemetry_20260827_144335.csv --output plan.csv --interval_ms 500

出力(plan.csv)の列:
    keyframe_id, time, image_path, cum_north_m, cum_east_m, cum_distance_m,
    alt, pitch, roll, yaw, stick_left_v, stick_left_h, stick_right_v, stick_right_h

    cum_north_m / cum_east_m : 教示開始地点からの推定相対変位 (m, NED座標系)
    cum_distance_m           : 教示開始地点からの走行距離の累積 (m, 常に増加する「進捗」の目安)
"""

import argparse
import math

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Build a Teach & Repeat plan from a synced telemetry CSV.")
    parser.add_argument("--telemetry", required=True, help="Path to telemetry_<sessionId>.csv (image_path列を含むもの)")
    parser.add_argument("--output", default="plan.csv", help="Output plan CSV path")
    parser.add_argument("--interval_ms", type=int, default=500, help="基準フレームの間引き間隔(ms)。デフォルト500ms")
    parser.add_argument(
        "--stall_speed_mps", type=float, default=0.05,
        help="この速度(m/s)未満を'止まっている'とみなす閾値。デフォルト0.05",
    )
    parser.add_argument(
        "--trailing_stall_min_s", type=float, default=5.0,
        help="末尾でこの秒数以上連続で止まっていたら、その区間をPlanから打ち切る。デフォルト5.0秒",
    )
    parser.add_argument("--ref_session_id", type=int, default=None, help="指定した場合、このsession_idの行のみからPlanを構築する")
    return parser.parse_args()


def trim_leading_stall(valid: pd.DataFrame, stick_th: int = 10) -> pd.DataFrame:
    """先頭から連続するスティック中立(操作なし)区間を除外する。
    教示開始直後に離陸待ちや静止している区間をPlanから省き、
    スティック操作(前進等)が始まった瞬間からPlanを開始させる。
    """
    sl_v = valid["stick_left_v"].abs().to_numpy() if "stick_left_v" in valid.columns else [0] * len(valid)
    sl_h = valid["stick_left_h"].abs().to_numpy() if "stick_left_h" in valid.columns else [0] * len(valid)
    sr_v = valid["stick_right_v"].abs().to_numpy() if "stick_right_v" in valid.columns else [0] * len(valid)
    sr_h = valid["stick_right_h"].abs().to_numpy() if "stick_right_h" in valid.columns else [0] * len(valid)

    first_action_idx = 0
    for i in range(len(valid)):
        if sl_v[i] >= stick_th or sl_h[i] >= stick_th or sr_v[i] >= stick_th or sr_h[i] >= stick_th:
            first_action_idx = i
            break

    if first_action_idx > 0:
        print(f"先頭の操作なし待機区間 (行 0〜{first_action_idx - 1}) をPlanから除外しました。")
        return valid.iloc[first_action_idx:].reset_index(drop=True)
    return valid


def trim_trailing_stall(valid: pd.DataFrame, stall_speed_mps: float, trailing_stall_min_s: float) -> pd.DataFrame:
    """末尾から連続する低速区間を検出し、それより前で打ち切る。"""
    vx = valid["vx"].to_numpy()
    vy = valid["vy"].to_numpy()
    speed = [math.hypot(vx[i], vy[i]) for i in range(len(valid))]
    times = valid["time"].to_numpy()

    # 末尾から走査し、最後に速度が閾値を超えていた行(=最後に動いていた瞬間)を探す
    last_moving_idx = None
    for i in range(len(valid) - 1, -1, -1):
        if speed[i] >= stall_speed_mps:
            last_moving_idx = i
            break

    if last_moving_idx is None or last_moving_idx == len(valid) - 1:
        # 全区間止まっている、または末尾が止まっていない(=打ち切り不要)
        return valid

    trailing_duration_s = (times[-1] - times[last_moving_idx]) / 1000.0
    if trailing_duration_s < trailing_stall_min_s:
        # 末尾の停止が短時間(教示走行の最後の減速など)なら打ち切らない
        return valid

    print(f"末尾に{trailing_duration_s:.1f}秒の停止区間を検出したため、打ち切ります"
          f"(行 {last_moving_idx+1}〜{len(valid)-1} を除外)")
    return valid.iloc[: last_moving_idx + 1].reset_index(drop=True)


def build_plan(df: pd.DataFrame, interval_ms: int, stall_speed_mps: float, trailing_stall_min_s: float) -> pd.DataFrame:
    # 1. image_pathが空の行(記録開始直後、カメラ初期化前など)を除外
    valid = df[df["image_path"].notna() & (df["image_path"].astype(str).str.strip() != "")].copy()
    valid = valid.sort_values("time").reset_index(drop=True)

    if len(valid) == 0:
        raise ValueError("image_pathが有効な行が1件もありません。")

    # NEW: 先頭の操作なし待機区間を除外し、末尾の停止区間を打ち切る
    valid = trim_leading_stall(valid)
    valid = trim_trailing_stall(valid, stall_speed_mps, trailing_stall_min_s)

    # 2. 相対変位の累積計算 (全有効行に対して、区間ごとの速度×dtを積算)
    times = valid["time"].to_numpy()
    vx = valid["vx"].to_numpy()  # NED: north
    vy = valid["vy"].to_numpy()  # NED: east

    cum_north = [0.0]
    cum_east = [0.0]
    cum_dist = [0.0]
    for i in range(1, len(valid)):
        dt = (times[i] - times[i - 1]) / 1000.0  # ms -> s
        if dt < 0:
            dt = 0.0
        # 区間開始時点の速度を使った矩形近似で積分
        dn = vx[i - 1] * dt
        de = vy[i - 1] * dt
        cum_north.append(cum_north[-1] + dn)
        cum_east.append(cum_east[-1] + de)
        cum_dist.append(cum_dist[-1] + math.hypot(dn, de))

    valid["cum_north_m"] = cum_north
    valid["cum_east_m"] = cum_east
    valid["cum_distance_m"] = cum_dist

    # 3. interval_msごとに間引いて基準フレームを抽出
    keyframe_rows = []
    last_time = None
    for _, row in valid.iterrows():
        if last_time is None or (row["time"] - last_time) >= interval_ms:
            keyframe_rows.append(row)
            last_time = row["time"]

    plan = pd.DataFrame(keyframe_rows).reset_index(drop=True)
    plan.insert(0, "keyframe_id", range(len(plan)))

    # 4. Yaw 角度の初期相対化 ＆ 移動平均フィルタ (手ブレ・カクつきの自動平滑化)
    if "yaw" in plan.columns and len(plan) > 0:
        raw_yaws = plan["yaw"].to_numpy()
        start_yaw = raw_yaws[0]

        # スタート時の向きを 0° とした相対角度に補正
        rel_yaws = []
        for y in raw_yaws:
            diff = y - start_yaw
            while diff > 180.0: diff -= 360.0
            while diff < -180.0: diff += 360.0
            rel_yaws.append(diff)

        # 移動平均フィルタ (5コマ平滑化) で手動操縦のブレを滑らかな曲線に変換
        smooth_yaws = []
        window = 5
        for i in range(len(rel_yaws)):
            start_i = max(0, i - window // 2)
            end_i = min(len(rel_yaws), i + window // 2 + 1)
            avg_yaw = sum(rel_yaws[start_i:end_i]) / (end_i - start_i)
            smooth_yaws.append(avg_yaw)

        plan["yaw"] = smooth_yaws

    # スティック列が0でsend_stick_*列またはcalc_stick_*列が存在する場合、自動で採用
    stick_map = {
        "stick_left_v": ["send_stick_left_v", "calc_stick_left_v"],
        "stick_left_h": ["send_stick_left_h", "calc_stick_left_h"],
        "stick_right_v": ["send_stick_right_v", "calc_stick_right_v"],
        "stick_right_h": ["send_stick_right_h", "calc_stick_right_h"],
    }
    for orig, candidates in stick_map.items():
        if orig not in plan.columns or (plan[orig] == 0).all():
            for cand in candidates:
                if cand in plan.columns and not (plan[cand] == 0).all():
                    plan[orig] = plan[cand]
                    break

    cols = [
        "keyframe_id", "time", "image_path",
        "cum_north_m", "cum_east_m", "cum_distance_m",
        "alt", "pitch", "roll", "yaw",
        "stick_left_v", "stick_left_h", "stick_right_v", "stick_right_h",
        "calc_stick_left_v", "calc_stick_left_h", "calc_stick_right_v", "calc_stick_right_h",
        "send_stick_left_v", "send_stick_left_h", "send_stick_right_v", "send_stick_right_h",
    ]
    plan = plan[[c for c in cols if c in plan.columns]]
    return plan


def main():
    args = parse_args()
    df = pd.read_csv(args.telemetry)

    if args.ref_session_id is not None and "session_id" in df.columns:
        print(f"session_id == {args.ref_session_id} のデータのみ抽出します。")
        df = df[df["session_id"] == args.ref_session_id].copy()

    total_rows = len(df)
    empty_image = df["image_path"].isna().sum() + (df["image_path"].astype(str).str.strip() == "").sum()
    print(f"telemetry総行数: {total_rows}")
    print(f"image_pathが空の行(除外対象): {empty_image}")

    plan = build_plan(df, args.interval_ms, args.stall_speed_mps, args.trailing_stall_min_s)

    print(f"基準フレーム数(間引き後): {len(plan)}")
    print(f"総走行距離(推定): {plan['cum_distance_m'].iloc[-1]:.2f} m")

    plan.to_csv(args.output, index=False)
    print(f"\nPlanを書き出しました: {args.output}")


if __name__ == "__main__":
    main()
