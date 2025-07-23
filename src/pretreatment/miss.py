import os
import glob

from config import setup_directories
from CreateHistogram import CreateHistogram_rg_gb

def miss():
    # ディレクトリ設定
    dirs = setup_directories()

    # MASKディレクトリから画像ファイル（.png）を取得
    mask_image_paths = sorted(glob.glob(os.path.join(dirs["MASK"], "*.png")))

    for image_path in mask_image_paths:
        try:
            CreateHistogram_rg_gb(image_path, dirs["TEMP"])
        except Exception as e:
            print(f"エラー: {image_path} の処理中に例外が発生しました: {e}")

# 実行
if __name__ == "__main__":
    miss()