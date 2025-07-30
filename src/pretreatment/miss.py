import os
import glob

from config import setup_directories
from CreateHistogram import CreateHistogram, CreateHistogram_rg_gb
       
def pretreatment():
    # ディレクトリ設定
    dirs = setup_directories()

    # png画像のパスを取得
    image_paths = sorted(glob.glob(os.path.join(dirs["INPUT"], "*.png")))

    
    for image_path in image_paths:
        try:
            # image_masked_path(マスク処理された画像)を使い,ヒストグラム(CSV)を作成,histpreディレクトリに保存
            CreateHistogram(image_path, dirs["TEMP"])
            CreateHistogram_rg_gb(image_path, dirs["TEMP"])
        except Exception as e:
            print(f"処理中にエラーが発生しました: {e}")

# 実行
if __name__ == "__main__":
    pretreatment()
