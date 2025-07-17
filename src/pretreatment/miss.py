import os
import glob

from config import setup_directories, BLACK_LEVEL
from CreateHistogram import CreateHistogram, CreateHistogram_rg_gb

def miss():
    # ディレクトリ設定
    dirs = setup_directories()

    # MASKディレクトリから画像ファイル（.png）を取得
    mask_image_paths = sorted(glob.glob(os.path.join(dirs["MASK"], "*.png")))

    for image_path in mask_image_paths:
        try:
            # ファイル名取得
            filename = os.path.splitext(os.path.basename(image_path))[0]

            # 画像を読み込み、BLACK_LEVELを一律に引く（0未満は0にクリップ）
            import cv2
            import numpy as np
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            img = img - BLACK_LEVEL
            img = np.clip(img, 0, None)
            img = img.astype(np.uint16)  # 必要に応じて型を調整

            # 一時ファイルとして保存
            temp_path = os.path.join(dirs["TEMP"], f"{filename}.png")
            cv2.imwrite(temp_path, img)

            # ヒストグラム(CSV)を作成し、histディレクトリに保存
            CreateHistogram(temp_path, dirs["HIST"])
            # ヒストグラム(numpy)を作成し、hist_rg_gbディレクトリに保存
            CreateHistogram_rg_gb(temp_path, dirs["HIST_RG_GB"])

            # 一時ファイル削除
            os.remove(temp_path)

            print(f"Processed histogram for: {image_path}")
            

        except Exception as e:
            print(f"エラー: {image_path} の処理中に例外が発生しました: {e}")

# 実行
if __name__ == "__main__":
    miss()