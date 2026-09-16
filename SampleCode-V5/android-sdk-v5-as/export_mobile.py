"""
export_mobile.py
学習された自作2D-CNNモデル (map_matching_model.pt) を Android 端末で動く .ptl 形式にエクスポートする。
"""

import torch
from train_map_matching import MapMatchingPolicy


def main():
    print("自作 2D-CNN モデルを Android 用 (.ptl) にエクスポート中...")

    # 1. モデル構造の定義と学習済み重みのロード
    model = MapMatchingPolicy(history_len=5)
    checkpoint = torch.load("map_matching_model_new.pt", map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # 2. ダミー入力データの作成 (バッチサイズ1, RGB3ch, 224x224, 履歴20次元)
    dummy_curr = torch.randn(1, 3, 224, 224)
    dummy_ref = torch.randn(1, 3, 224, 224)
    dummy_hist = torch.randn(1, 20)

    # 3. TorchScript にトレース変換
    traced_script = torch.jit.trace(model, (dummy_curr, dummy_ref, dummy_hist))

    # 4. モバイル用最適化 (環境依存エラーを回避して安全に保存)
    try:
        from torch.utils.mobile_optimizer import optimize_for_mobile
        script_for_mobile = optimize_for_mobile(traced_script)
    except Exception as e:
        print(f"（注: mobile_optimizer をスキップし直接 LiteInterpreter 形式で保存します）")
        script_for_mobile = traced_script

    # 5. Android 用 .ptl ファイルとして保存
    output_ptl = "map_matching_model_mobile.ptl"
    script_for_mobile._save_for_lite_interpreter(output_ptl)
    print(f"✨ 成功! Android用モデルを出力しました: {output_ptl}")


if __name__ == "__main__":
    main()
