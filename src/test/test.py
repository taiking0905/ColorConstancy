import rawpy
import numpy as np
import cv2

# 入力ファイル（Canon RAWファイル）
dng_path = './IMG_3891.DNG'

with rawpy.imread(dng_path) as raw:
    try:
        as_shot_neutral = np.array(raw.camera_whitebalance, dtype=np.float32)
        if as_shot_neutral is None or np.any(as_shot_neutral == 0):
            raise ValueError
    except:
        as_shot_neutral = np.array(raw.daylight_whitebalance, dtype=np.float32)

    # RGBの最初の3要素だけを使用（4要素ある場合もあるため）
    as_shot_neutral = as_shot_neutral[:3]
    as_shot_neutral[as_shot_neutral == 0] = 1e-6  # 0除算対策

    wb_multipliers = 1.0 / as_shot_neutral

    print("AsShotNeutral or fallback (used):", as_shot_neutral)
    print("WB multipliers:", wb_multipliers)


with rawpy.imread(dng_path) as raw:
    # デモザイク後のRGBを16bitで取得（ホワイトバランスなし・ガンマ補正なし）
    rgb = raw.postprocess(
        use_camera_wb=False,       # ホワイトバランス無効
        no_auto_bright=True,       # 自動明るさ調整オフ
        output_bps=16,             # 16bit出力
        gamma=(1, 1),              # ガンマ補正なし（リニア）
        demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD  # デモザイクあり（AHDなど）
    )

    # データの最大値（RAWセンサーの白レベル）を取得
    white_level = raw.white_level
    black_level = raw.black_level_per_channel[0]  # 通常全チャンネル同じ
    print("white_level:", white_level, "black_level", black_level)

    # black level 補正
    # rgb = np.clip(rgb.astype(np.float32) - black_level, 0, white_level - black_level)

    # 12bitに正規化（値域0〜4095へスケーリング）
    rgb_12bit = (rgb / (white_level - black_level) * 4095).astype(np.uint16)

    # 保存（16bit PNG、値域は0〜4095の12bit相当）
    cv2.imwrite("IMG_3891.png", cv2.cvtColor(rgb_12bit, cv2.COLOR_RGB2BGR))
