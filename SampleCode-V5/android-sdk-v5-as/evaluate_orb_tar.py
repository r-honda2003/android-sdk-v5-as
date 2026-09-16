import argparse
import os
import cv2
import numpy as np
import pandas as pd
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Geometric ORB-TaR on a dataset.")
    parser.add_argument("--telemetry", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--val_session_id", type=int, default=-1)
    parser.add_argument("--num_examples", type=int, default=10)
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.telemetry)

    # Filter for validation session
    available_sessions = sorted(df["session_id"].unique())
    val_sid = args.val_session_id if args.val_session_id != -1 else available_sessions[-1]
    val_df = df[df["session_id"] == val_sid].reset_index(drop=True)

    print(f"Evaluating ORB-TaR on Session {val_sid} ({len(val_df)} samples)")

    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Constants for stick calculation (Same logic as proposed for Android)
    ROLL_GAIN = 0.5
    PITCH_SPEED = 100

    results = []

    for i in range(len(val_df)):
        row = val_df.iloc[i]

        curr_path = os.path.join(args.images_dir, row["image_path"])
        ref_path = os.path.join(args.images_dir, row["ref_image_path"])

        img_curr = cv2.imread(curr_path, cv2.IMREAD_GRAYSCALE)
        img_ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)

        if img_curr is None or img_ref is None:
            continue

        # Detect and Compute
        kp1, des1 = orb.detectAndCompute(img_curr, None)
        kp2, des2 = orb.detectAndCompute(img_ref, None)

        pred_roll = 0

        if des1 is not None and des2 is not None:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)[:50]

            if len(matches) >= 10:
                dxs = []
                for m in matches:
                    pt_curr = kp1[m.queryIdx].pt
                    pt_ref = kp2[m.trainIdx].pt
                    dxs.append(pt_curr[0] - pt_ref[0])

                avg_dx = np.mean(dxs)
                norm_dx = avg_dx / img_curr.shape[1] # Normalized by width

                # Simple P-control for Roll
                # If features shifted right (dx > 0), drone is too far left?
                # Need to match drone's coordinate system.
                pred_roll = int(-norm_dx * 1000 * ROLL_GAIN)
                pred_roll = max(-300, min(300, pred_roll))

        actual_roll = row["stick_right_h"]
        error = abs(pred_roll - actual_roll)
        results.append(error)

        if i < args.num_examples:
            print(f"[{i:3d}] Actual Roll: {actual_roll:4d} | Pred Roll: {pred_roll:4d} | Error: {error:4d}")

    if results:
        print(f"\nMean Absolute Error (Roll): {np.mean(results):.1f}")
    else:
        print("No matches found for any frame.")

if __name__ == "__main__":
    main()
