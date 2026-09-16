import pandas as pd

def main():
    df = pd.read_csv("matched_telemetry_new.csv")

    # Session 2 is our reference map
    ref_df = df[df["session_id"] == 2].copy()

    # Android expects:
    # keyframe_id, time, image_path, cum_north_m, cum_east_m, cum_distance_m, alt, pitch, roll, yaw, stick_left_v, stick_left_h, stick_right_v, stick_right_h

    # Map matched_telemetry_new.csv columns to plan.csv columns
    plan_df = pd.DataFrame()
    plan_df["keyframe_id"] = range(len(ref_df))
    plan_df["time"] = ref_df["time"].values
    plan_df["image_path"] = ref_df["image_path"].values
    plan_df["cum_north_m"] = 0.0 # Not used by current VM but needed for format
    plan_df["cum_east_m"] = 0.0  # Not used by current VM but needed for format
    plan_df["cum_distance_m"] = ref_df["cum_dist_m"].values
    plan_df["alt"] = ref_df["alt"].values
    plan_df["pitch"] = ref_df["pitch"].values
    plan_df["roll"] = ref_df["roll"].values
    plan_df["yaw"] = ref_df["yaw"].values
    plan_df["stick_left_v"] = ref_df["stick_left_v"].values
    plan_df["stick_left_h"] = ref_df["stick_left_h"].values
    plan_df["stick_right_v"] = ref_df["stick_right_v"].values
    plan_df["stick_right_h"] = ref_df["stick_right_h"].values

    plan_df.to_csv("plan_new.csv", index=False)
    print("Created plan_new.csv for Android assets.")

if __name__ == "__main__":
    main()
