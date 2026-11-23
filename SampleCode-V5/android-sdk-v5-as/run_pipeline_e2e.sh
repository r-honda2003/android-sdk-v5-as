#!/usr/bin/env bash
set -e

# 環境変数の設定 (NumPy 2.x 互換性警告の抑制)
export BOTTLENECK_DISABLE=1
export NUMEXPR_MAX_THREADS=1

# パラメータ設定 (コマンドライン引数で上書き可能)
# 使い方: ./run_pipeline_e2e.sh [TELEMETRY_CSV] [IMAGES_DIR] [EPOCHS] [EXCLUDE_SESSIONS] [RUN_PREFIX]
TELEMETRY_CSV="${1:-telemetry_with_sessions.csv}"
IMAGES_DIR="${2:-images_run1}"
EPOCHS="${3:-10}"
EXCLUDE_SESSIONS="${4:-}"
RUN_PREFIX="${5:-run1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 0. 端末からの自動取得＆統合 (分割記録されたセッションの自動マージ)
echo -e "\n[Step 0] Pulling & Merging sessions ($RUN_PREFIX) from device if available ..."
python3 pull_and_merge_sessions.py --run_prefix "$RUN_PREFIX" --output_csv "$TELEMETRY_CSV" --output_img_dir "$IMAGES_DIR" || true

EXCLUDE_ARG=""
if [ -n "$EXCLUDE_SESSIONS" ]; then
    EXCLUDE_ARG="--exclude_sessions $EXCLUDE_SESSIONS"
fi

echo "=========================================="
echo " Pure E2E Pipeline Execution Configuration:"
echo "   TELEMETRY_CSV    : $TELEMETRY_CSV"
echo "   IMAGES_DIR       : $IMAGES_DIR"
echo "   EPOCHS           : $EPOCHS"
echo "   EXCLUDE_SESSIONS : ${EXCLUDE_SESSIONS:-None (All 1-5 sessions included)}"
echo "=========================================="

# 0.5. 入力チェック
if [ ! -f "$TELEMETRY_CSV" ]; then
    echo -e "\n[ERROR] $TELEMETRY_CSV が存在しません。"
    echo "指定したテレメトリCSVファイル名を確認してください。"
    exit 1
fi

if [ ! -d "$IMAGES_DIR" ]; then
    echo -e "\n[ERROR] 画像ディレクトリ $IMAGES_DIR が存在しません。"
    echo "指定した画像ディレクトリ名を確認してください。"
    exit 1
fi

# 1. セッション分割 (split_sessions.py)
echo -e "\n[Step 1] Splitting sessions from $TELEMETRY_CSV ..."
python3 split_sessions.py --telemetry "$TELEMETRY_CSV" --output telemetry_with_sessions.csv

# 2. 純粋E2Eベースラインモデルの学習 (train_baseline.py)
echo -e "\n[Step 2] Training Pure E2E Baseline Model with IMAGES_DIR=$IMAGES_DIR (epochs=$EPOCHS) ..."
python3 train_baseline.py --telemetry telemetry_with_sessions.csv --images_dir "$IMAGES_DIR" --output_model baseline_model.pt --epochs "$EPOCHS" $EXCLUDE_ARG

# 3. PyTorch Mobile (TorchScript) 形式へのエクスポート & アセットコピー
ASSET_DIR="../android-sdk-v5-sample/src/main/assets"
if [ -f "baseline_model.pt" ]; then
    echo -e "\n[Step 3] Exporting Pure E2E TorchScript model for Android PyTorch Mobile ..."
    python3 export_mobile_model.py --model baseline_model.pt --output map_matching_model_android.pt --type baseline

    # コース総距離・進捗判定用の plan.csv を生成してアセットへコピー
    echo -e "\nGenerating plan.csv from $TELEMETRY_CSV for route progress and auto-stop..."
    python3 build_plan.py --telemetry "$TELEMETRY_CSV" --output plan.csv

    if [ -f "plan.csv" ] && [ -d "$ASSET_DIR" ]; then
        echo "Copying plan.csv to $ASSET_DIR/plan.csv ..."
        cp plan.csv "$ASSET_DIR/plan.csv"
    fi

    if [ -d "$ASSET_DIR" ]; then
        echo "Copying map_matching_model_android.pt to $ASSET_DIR/map_matching_model_android.pt ..."
        cp map_matching_model_android.pt "$ASSET_DIR/map_matching_model_android.pt"
    fi
fi

# 4. アプリビルド ＆ デバイスインストール
echo -e "\n[Step 4] Building and Installing Android App..."
if [ -x "./gradlew" ]; then
    ./gradlew sample:assembleDebug
    if [ -f "../android-sdk-v5-sample/build/outputs/apk/debug/sample-debug.apk" ]; then
        echo "Installing APK via adb..."
        adb install -r ../android-sdk-v5-sample/build/outputs/apk/debug/sample-debug.apk
    fi
fi

echo -e "\n=========================================="
echo " Pure E2E Pipeline Finished Successfully!"
echo " (Note: No plan.csv or image folder transfer required on device)"
echo "=========================================="
