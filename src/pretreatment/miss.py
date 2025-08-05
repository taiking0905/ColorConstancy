# import os
# import glob

# from config import setup_directories
# from CreateHistogram import CreateHistogram, CreateHistogram_rg_gb
       
# def pretreatment():
#     # ディレクトリ設定
#     dirs = setup_directories()

#     # png画像のパスを取得
#     image_paths = sorted(glob.glob(os.path.join(dirs["INPUT"], "*.png")))

    
#     for image_path in image_paths:
#         try:
#             # image_masked_path(マスク処理された画像)を使い,ヒストグラム(CSV)を作成,histpreディレクトリに保存
#             CreateHistogram(image_path, dirs["TEMP"])
#             CreateHistogram_rg_gb(image_path, dirs["TEMP"])
#         except Exception as e:
#             print(f"処理中にエラーが発生しました: {e}")

# # 実行
# if __name__ == "__main__":
#     pretreatment()

import os
import glob
import shutil

from config import setup_directories
from CreateHistogram import CreateHistogram, CreateHistogram_rg_gb
from MaskProcessing import MaskProcessing
from AnalayzeWhite import analyze_white_patch

       
def pretreatment():
    # ディレクトリ設定
    dirs = setup_directories()

    # png画像のパスを取得
    image_paths = sorted(glob.glob(os.path.join(dirs["END"], "*.png")))

    
    for image_path in image_paths:
        try:
            # 画像ファイル名から拡張子を除去して、マスクと教師データのパスを設定
            filename = os.path.splitext(os.path.basename(image_path))[0]
            # マスク処理した画像の名前を設定
            image_masked_path = os.path.join(dirs["MASK"], f"{filename}_masked.png")

            # 教師データの画像の名前を設定
            # image_corrected_path = os.path.join(dirs["TEACHER"], f"{filename}_corrected.png")
            # データ作成 エラーが出るなら止める
    
            # image_masked_path(マスク処理された画像)を使い,ヒストグラム(CSV)を作成,histpreディレクトリに保存
            CreateHistogram(image_masked_path, dirs["HIST"])
            CreateHistogram_rg_gb(image_masked_path, dirs["HIST_RG_GB"])

        except Exception as e:
            print(f"処理中にエラーが発生しました: {e}")

# 実行
if __name__ == "__main__":
    pretreatment()
