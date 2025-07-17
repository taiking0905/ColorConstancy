import os
import glob

from config import setup_directories, BLACK_LEVEL

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
            img = img.astype(np.uint16)

            # TEMPディレクトリに保存
            temp_path = os.path.join(dirs["TEMP"], f"{filename}.png")
            cv2.imwrite(temp_path, img)

            print(f"Saved to TEMP: {temp_path}")

        except Exception as e:
            print(f"エラー: {image_path} の処理中に例外が発生しました: {e}")

# 実行
if __name__ == "__main__":
    miss()