"""
extract_and_split_mp4.py

1本のMP4動画（着陸・静止・離陸・走行が繰り返される動画）から、
「実際に走っている（飛行移動している）区間」を動き検知（フレーム差分）で自動検出し、
着陸中・停止中を除外して周ごとに session_id (1〜N) を振り分けて画像と telemetry_with_sessions.csv を自動生成します。

使い方:
    1. 動き検知で走っている区間を完全自動検出する場合:
       python3 extract_and_split_mp4.py --video teach_5runs.mp4 --auto_detect --output_dir images_5runs --output_csv telemetry_with_sessions.csv

    2. 時間(秒)を明示指定して切り出す場合:
       python3 extract_and_split_mp4.py --video teach_5runs.mp4 --splits "10-25, 30-45, 50-65, 70-85, 90-105" --output_dir images_5runs --output_csv telemetry_with_sessions.csv

オプション:
    --video        : 入力MP4動画のパス (必須)
    --auto_detect  : フレーム差分による「走っている区間」の自動検出モード
    --motion_th    : 動き検知の感度閾値 (デフォルト: 5.0。低いほど敏感、高いほど鈍感)
    --num_sessions : セッションの数 (デフォルト: 5)
    --splits       : 時間指定 (例: "10-25, 30-45")
    --interval_sec : フレーム切り出し間隔(秒) (デフォルト: 0.1)
    --output_dir   : 画像保存フォルダ名 (デフォルト: images_5runs)
    --output_csv   : 出力CSV名 (デフォルト: telemetry_with_sessions.csv)
"""

import argparse
import os
import cv2
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-detect active flight runs from MP4 video using motion detection.")
    parser.add_argument("--video", required=True, help="Input MP4 video file path")
    parser.add_argument("--auto_detect", action="store_true", default=True, help="Auto-detect running sections via motion analysis (default: True)")
    parser.add_argument("--motion_th", type=float, default=5.0, help="Motion detection threshold (default: 5.0)")
    parser.add_argument("--num_sessions", type=int, default=5, help="Expected number of sessions (default: 5)")
    parser.add_argument("--splits", type=str, default="", help="Optional manual start-end times in seconds (e.g. '10-25, 30-45')")
    parser.add_argument("--interval_sec", type=float, default=0.1, help="Extraction interval in seconds (default: 0.1)")
    parser.add_argument("--output_dir", default="images_5runs", help="Output directory for images")
    parser.add_argument("--output_csv", default="telemetry_with_sessions.csv", help="Output CSV path")
    parser.add_argument("--speed_mps", type=float, default=1.0, help="Estimated flight speed in m/s (default: 1.0)")
    return parser.parse_args()


def detect_active_sessions_by_motion(video_path, motion_threshold=5.0, min_session_sec=2.0, min_idle_sec=1.5):
    """動画のフレーム差分を解析し、走っている（移動中）区間だけを自動検出する"""
    print("\n[自動検出] 動画のカメラ移動(フレーム差分)を解析中...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, prev = cap.read()
    if not ret:
        return []

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    frame_step = max(1, int(fps * 0.1))  # Check motion every ~0.1 second

    timestamps = []
    motion_scores = []

    curr_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        curr_frame += 1
        if curr_frame % frame_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = float(np.mean(cv2.absdiff(prev_gray, gray)))
            time_sec = curr_frame / fps
            timestamps.append(time_sec)
            motion_scores.append(diff)
            prev_gray = gray

    cap.release()

    if not motion_scores:
        return []

    # Smooth motion scores using moving average
    kernel_size = 5
    smoothed = np.convolve(motion_scores, np.ones(kernel_size)/kernel_size, mode='same')

    # Detect active intervals
    is_active = [score >= motion_threshold for score in smoothed]

    ranges = []
    in_active = False
    start_time = 0.0

    for i in range(len(timestamps)):
        t = timestamps[i]
        active = is_active[i]

        if not in_active and active:
            in_active = True
            start_time = t
        elif in_active and not active:
            # Check if duration is long enough
            if (t - start_time) >= min_session_sec:
                ranges.append((start_time, t))
            in_active = False

    if in_active:
        end_time = timestamps[-1]
        if (end_time - start_time) >= min_session_sec:
            ranges.append((start_time, end_time))

    return ranges


def parse_manual_splits(splits_str):
    ranges = []
    parts = splits_str.split(",")
    for p in parts:
        p = p.strip()
        if "-" in p:
            start, end = p.split("-")
            ranges.append((float(start), float(end)))
    return ranges


def main():
    args = parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video '{args.video}'.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0
    cap.release()

    print(f"Video loaded: '{args.video}'")
    print(f"FPS: {fps:.2f}, Total frames: {total_frames}, Duration: {duration_sec:.2f}s")

    if args.splits.strip():
        print("[手動指定] 指定された時間範囲を使用します...")
        ranges = parse_manual_splits(args.splits)
    else:
        ranges = detect_active_sessions_by_motion(args.video, motion_threshold=args.motion_th)
        if not ranges:
            print("Warning: 自動検出で飛行区間が見つかりませんでした。動画全体を均等分割します。")
            segment_len = duration_sec / args.num_sessions
            ranges = [(i * segment_len, (i + 1) * segment_len) for i in range(args.num_sessions)]

    print(f"\n================ 検出された「走っている」セッション一覧 ({len(ranges)}周分) ================")
    for idx, (s_start, s_end) in enumerate(ranges, start=1):
        dur = s_end - s_start
        print(f"  Session {idx}:  {s_start:6.2f}s  ->  {s_end:6.2f}s  (長さ: {dur:5.2f}s)")
    print("==================================================================================")

    # Process and extract frames
    cap = cv2.VideoCapture(args.video)
    rows = []
    base_timestamp = 1700000000000  # Base epoch timestamp

    frame_step = max(1, int(fps * args.interval_sec))

    for session_idx, (s_start, s_end) in enumerate(ranges, start=1):
        start_frame = int(s_start * fps)
        end_frame = int(s_end * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        curr_frame_idx = start_frame

        while curr_frame_idx <= end_frame and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if (curr_frame_idx - start_frame) % frame_step == 0:
                rel_sec = (curr_frame_idx - start_frame) / fps
                time_ms = base_timestamp + session_idx * 100000000 + int(rel_sec * 1000)
                image_name = f"{time_ms}.jpg"
                image_path = os.path.join(args.output_dir, image_name)

                # Save frame image
                cv2.imwrite(image_path, frame)

                rows.append({
                    "time": time_ms,
                    "image_path": image_name,
                    "vx": args.speed_mps,
                    "vy": 0.0,
                    "vz": 0.0,
                    "pitch": 0.0,
                    "roll": 0.0,
                    "yaw": 0.0,
                    "lat": 0.0,
                    "lng": 0.0,
                    "alt": 1.0,
                    "gps_signal": 5,
                    "battery_percent": 100,
                    "stick_left_v": 0,
                    "stick_left_h": 0,
                    "stick_right_v": 120, # Gentle forward stick
                    "stick_right_h": 0,
                    "session_id": session_idx
                })

            curr_frame_idx += 1

    cap.release()

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)

    print(f"\n[完了] '{args.output_dir}/' に {len(rows)} 枚の画像を保存しました。")
    print(f"[完了] 分割済みCSV: '{args.output_csv}' を生成しました。")


if __name__ == "__main__":
    main()
