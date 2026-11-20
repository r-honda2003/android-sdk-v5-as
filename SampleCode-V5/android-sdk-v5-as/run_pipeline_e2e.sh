#!/usr/bin/env bash
set -e

# 環境変数の設定 (NumPy 2.x 互換性警告の抑制)
export BOTTLENECK_DISABLE=1
export NUMEXPR_MAX_THREADS=1

# パラメータ設定 (コマンドライン引数で上書き可能)
# 使い方: ./run_pipeline_e2e.sh <TELEMETRY_CSV> <IMAGES_DIR> [EPOCHS]
TELEMETRY_CSV="${1:-telemetry_20261120_092631.csv}"
IMAGES_DIR="${2:-images_20261120_092631}"
EPOCHS="${3:-10}"

echo "=========================================="
echo " Pure E2E Pipeline Execution Configuration:"
echo "   TELEMETRY_CSV : $TELEMETRY_CSV"
echo "   IMAGES_DIR    : $IMAGES_DIR"
echo "   EPOCHS        : $EPOCHS"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 0. 入力チェック
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
python3 train_baseline.py --telemetry telemetry_with_sessions.csv --images_dir "$IMAGES_DIR" --output_model baseline_model.pt --epochs "$EPOCHS"

# 3. PyTorch Mobile (TorchScript) 形式へのエクスポート & アセットコピー
ASSET_DIR="../android-sdk-v5-sample/src/main/assets"
if [ -f "baseline_model.pt" ]; then
    echo -e "\n[Step 3] Exporting Pure E2E TorchScript model for Android PyTorch Mobile ..."
    python3 export_mobile_model.py --model baseline_model.pt --output map_matching_model_android.pt --type baseline

    # 古い plan.csv があれば除去して Pure E2E モードを有効化
    if [ -f "$ASSET_DIR/plan.csv" ]; then
        echo "Removing old plan.csv to activate Pure E2E Mode on Android..."
        rm -f "$ASSET_DIR/plan.csv"
    fi

    if [ -d "$ASSET_DIR" ]; then
        echo "Copying map_matching_model_android.pt to $ASSET_DIR/map_matching_model_android.pt ..."
        cp map_matching_model_android.pt "$ASSET_DIR/map_matching_model_android.pt"
    fi
fi

# 4. アプリビルド ＆ デバイスインストール
echo -e "\n[Step 4] Building and Installing Android App..."
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
if [ -x "./gradlew" ]; then
    ./gradlew sample:installDebug
fi

echo -e "\n=========================================="
echo " Pure E2E Pipeline Finished Successfully!"
echo " (Note: No plan.csv or image folder transfer required on device)"
echo "=========================================="
