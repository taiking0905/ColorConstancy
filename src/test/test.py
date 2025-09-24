import rawpy
import numpy as np
import cv2
import glob
import os
from dotenv import load_dotenv

load_dotenv()
# DNGファイルが入っているディレクトリ
TEST_IPHONE_PATH = os.getenv("TEST_IPHONE_PATH")

# ディレクトリ内のすべてのDNGファイルを取得
dng_files = glob.glob(os.path.join(TEST_IPHONE_PATH, "*.DNG"))

for dng_path in dng_files:
    try:
        with rawpy.imread(dng_path) as raw:
            # WB情報の取得（camera_whitebalance または daylight_whitebalance）
            try:
                as_shot_neutral = np.array(raw.camera_whitebalance, dtype=np.float32)
                if as_shot_neutral is None or np.any(as_shot_neutral == 0):
                    raise ValueError
            except:
                as_shot_neutral = np.array(raw.daylight_whitebalance, dtype=np.float32)

            as_shot_neutral = as_shot_neutral[:3]  # 最初の3要素を使用
            as_shot_neutral[as_shot_neutral == 0] = 1e-6
            wb_multipliers = 1.0 / as_shot_neutral

            # デモザイク後のRGBを16bitで取得（ホワイトバランスなし・ガンマ補正なし）
            rgb = raw.postprocess(
                use_camera_wb=False,
                no_auto_bright=True,
                no_auto_scale=True,
                output_bps=16,
                gamma=(1, 1),
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                #学習用はRAW,評価用はXYZ
                output_color=rawpy.ColorSpace.raw
            )

            # データの最大値（RAWセンサーの白レベル）を取得
            white_level = raw.white_level
            black_level = raw.black_level_per_channel[0]
            raw_image = raw.raw_image.copy()  
            min_val = raw_image.min()

            # 保存ファイル名（拡張子をPNGに変更）
            filename = os.path.splitext(os.path.basename(dng_path))[0] + ".png"
            save_path = os.path.join(TEST_IPHONE_PATH, filename)

            # PNGとして保存
            cv2.imwrite(save_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            # ログ出力
            print(f"\nファイル: {dng_path}")
            print("  AsShotNeutral:", as_shot_neutral)
            print("  WB multipliers:", wb_multipliers)
            print("  white_level:", white_level, "black_level:", black_level)
            print("  RAW最小値:", min_val)
            print(f"  保存しました: {save_path}")

    except Exception as e:
        print(f"処理中にエラーが発生しました ({dng_path}): {e}")
