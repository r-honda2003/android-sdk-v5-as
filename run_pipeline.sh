#!/bin/bash
# =========================================================
# 自動学習 ＆ 反映一括実行スクリプト (run_pipeline.sh)
# =========================================================

PROJECT_ROOT="/home/ruri/AndroidStudioProjects/Mobile-SDK-Android-V5"
AS_DIR="$PROJECT_ROOT/SampleCode-V5/android-sdk-v5-as"
ASSETS_DIR="$PROJECT_ROOT/SampleCode-V5/android-sdk-v5-sample/src/main/assets"
SAMPLE_DIR="$PROJECT_ROOT/SampleCode-V5"

# 引数またはデフォルトのファイル名
TELEMETRY_CSV="${1:-telemetry_20261118_113402.csv}"
IMAGES_DIR="${2:-images_20261118_113402}"

echo "=========================================="
echo "使用データ: $TELEMETRY_CSV / $IMAGES_DIR"
echo "=========================================="

# 1. Python処理 ＆ モデル学習
cd "$AS_DIR" || exit 1

echo "Step 1: セッション分割..."
python3 split_sessions.py --telemetry "$TELEMETRY_CSV" --output telemetry_with_sessions.csv || exit 1

echo "Step 2: 逆算スティック制御量計算..."
python3 compute_calculated_sticks.py --input telemetry_with_sessions.csv --output telemetry_with_calc.csv || exit 1

echo "Step 3: 地図(plan.csv)作成..."
python3 build_plan.py --telemetry telemetry_with_calc.csv --output plan.csv --interval_ms 500 || exit 1

echo "Step 4: マッチング紐付け..."
python3 match_sessions.py --telemetry telemetry_with_calc.csv --ref_session_id 1 --output matched_telemetry.csv || exit 1

echo "Step 5: PyTorchモデル学習..."
python3 train_map_matching.py \
    --telemetry matched_telemetry.csv \
    --images_dir "$IMAGES_DIR" \
    --output_model map_matching_model.pt \
    --epochs 5 \
    --val_session_id -1 \
    --history_len 5 || exit 1

echo "Step 6: Android用モデル変換..."
python3 -c "
import torch
import sys
sys.path.append('$AS_DIR')
from model_def import MapMatchingPolicy

history_len = 5
model = MapMatchingPolicy(history_len=history_len, freeze_backbone=True)
checkpoint = torch.load('map_matching_model.pt', map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

x = torch.rand(1, 3, 224, 224)
ref_x = torch.rand(1, 3, 224, 224)
hist = torch.zeros(1, history_len * 4)

traced = torch.jit.trace(model, (x, ref_x, hist))
traced._save_for_lite_interpreter('map_matching_model_android.pt')
print('Android用モデル変換完了: map_matching_model_android.pt')
" || exit 1

# 2. アプリ assets へのコピー ＆ 実機への画像転送
echo "Step 7: plan.csv とモデルのコピー..."
cp "$AS_DIR/plan.csv" "$ASSETS_DIR/plan.csv"
cp "$AS_DIR/map_matching_model_android.pt" "$ASSETS_DIR/map_matching_model_android.pt"

echo "Step 8: 画像の adb push..."
adb push "$AS_DIR/$IMAGES_DIR" /sdcard/Android/data/dji.sampleV5.aircraft/files/

# 3. アプリのビルド ＆ インストール
echo "Step 9: アプリのビルド・インストール..."
cd "$AS_DIR" || exit 1
./gradlew installDebug

echo "=========================================="
echo "すべての処理が正常に完了しました！"
echo "=========================================="
