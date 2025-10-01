import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import cv2

def png_to_jpeg(png_path, jpeg_path=None, quality=95):
    """
    PNG画像をJPEGに変換して保存する
    - png_path: 入力PNGの絶対パス
    - jpeg_path: 出力JPEGの絶対パス（Noneなら同じ場所に拡張子だけ変更）
    - quality: JPEG品質（0-100, デフォルト95）
    """
    # 入力画像を読み込み
    img = mpimg.imread(png_path)
    if img is None:
        raise FileNotFoundError(f"画像が読み込めませんでした: {png_path}")

    # 出力パスを決定
    if jpeg_path is None:
        jpeg_path = os.path.splitext(png_path)[0] + ".jpeg"

    # JPEGで保存
    success = cv2.imwrite(jpeg_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if success:
        print(f"JPEG保存成功: {jpeg_path}")
    else:
        print(f"JPEG保存失敗: {jpeg_path}")
    return jpeg_path


# ===== 使用例 =====
png_file = r"E:\ColorConstancy\enddatasets\8D5U5524.png" # 変換したいPNG
jpeg_file = r"E:\8D5U5524.jpeg"  # 保存先
png_to_jpeg(png_file, jpeg_file, quality=95)
