#!/usr/bin/env bash
set -e

# 環境変数の設定 (NumPy 2.x 互換性警告の抑制)
export BOTTLENECK_DISABLE=1
export NUMEXPR_MAX_THREADS=1

# パラメータ設定 (コマンドライン引数で上書き可能。デフォルト値は下記)
TELEMETRY_CSV="${1:-telemetry_with_sessions.csv}"
IMAGES_DIR="${2:-images_run1}"
REF_SESSION="${3:-1}"
EPOCHS="${4:-10}"
EXCLUDE_SESSIONS="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 0. 端末からの自動取得＆統合 (分割記録されたセッションの自動マージ)
echo -e "\n[Step 0] Pulling & Merging sessions from device if available ..."
python3 pull_and_merge_sessions.py --output_csv "$TELEMETRY_CSV" --output_img_dir "$IMAGES_DIR" || true

EXCLUDE_ARG=""
if [ -n "$EXCLUDE_SESSIONS" ]; then
    EXCLUDE_ARG="--exclude_sessions $EXCLUDE_SESSIONS"
fi

echo "=========================================="
echo " Pipeline Execution Configuration:"
echo "   TELEMETRY_CSV    : $TELEMETRY_CSV"
echo "   IMAGES_DIR       : $IMAGES_DIR"
echo "   REF_SESSION      : $REF_SESSION"
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

# 2. セッションマッチング (match_sessions.py)
echo -e "\n[Step 2] Matching sessions with REF_SESSION=$REF_SESSION ..."
python3 match_sessions.py --telemetry telemetry_with_sessions.csv --ref_session_id "$REF_SESSION" --output matched_telemetry.csv $EXCLUDE_ARG

# 3. Plan構築 (build_plan.py)
echo -e "\n[Step 3] Building plan from matched_telemetry.csv (REF_SESSION=$REF_SESSION) ..."
python3 build_plan.py --telemetry matched_telemetry.csv --ref_session_id "$REF_SESSION" --output plan.csv

# 4. plan.csv を Android アセットディレクトリへコピー
ASSET_DIR="../android-sdk-v5-sample/src/main/assets"
if [ -f "plan.csv" ] && [ -d "$ASSET_DIR" ]; then
    echo -e "\n[Step 4] Copying plan.csv to $ASSET_DIR/plan.csv ..."
    cp plan.csv "$ASSET_DIR/plan.csv"
fi

# 5. マップ照合モデルの学習 (train_map_matching.py)
echo -e "\n[Step 5] Training map matching model with IMAGES_DIR=$IMAGES_DIR (epochs=$EPOCHS) ..."
python3 train_map_matching.py --telemetry matched_telemetry.csv --images_dir "$IMAGES_DIR" --output_model map_matching_model.pt --epochs "$EPOCHS" $EXCLUDE_ARG

# 5.5. PyTorch Mobile (TorchScript) 形式へのエクスポート & アセットコピー
if [ -f "map_matching_model.pt" ]; then
    echo -e "\n[Step 5.5] Exporting TorchScript model for Android PyTorch Mobile ..."
    python3 export_mobile_model.py --model map_matching_model.pt --output map_matching_model_android.pt
    if [ -d "$ASSET_DIR" ]; then
        echo "Copying map_matching_model_android.pt to $ASSET_DIR/map_matching_model_android.pt ..."
        cp map_matching_model_android.pt "$ASSET_DIR/map_matching_model_android.pt"
    fi
fi

# 6. アプリビルド ＆ デバイスインストール
echo -e "\n[Step 6] Building and Installing Android App..."
if [ -x "./gradlew" ]; then
    ./gradlew sample:assembleDebug
    if [ -f "../android-sdk-v5-sample/build/outputs/apk/debug/sample-debug.apk" ]; then
        echo "Installing APK via adb..."
        adb install -r ../android-sdk-v5-sample/build/outputs/apk/debug/sample-debug.apk
    fi
fi

# 7. 画像ディレクトリをAndroid端末の外部ストレージ領域へ転送 (adb push)
cd "$SCRIPT_DIR"
echo -e "\n[Step 7] Pushing $IMAGES_DIR to Android device ..."
adb push "$IMAGES_DIR" /sdcard/Android/data/dji.sampleV5.aircraft/files/

echo -e "\n=========================================="
echo " Pipeline Finished Successfully!"
echo "=========================================="
