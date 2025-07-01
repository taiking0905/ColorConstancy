import os
import glob
import shutil 

from config import setup_directories
from CreateHistogram import CreateHistogram
from MaskProcessing import MaskProcessing
from TeacherProcessing import TeacherProcessing

       
def pretreatment():
    # ディレクトリ設定
    dirs = setup_directories()

    # png画像のパスを取得
    image_paths = sorted(glob.glob(os.path.join(dirs["INPUT"], "*.png")))

    
    for image_path in image_paths:
        try:
            # 画像ファイル名から拡張子を除去して、マスクと教師データのパスを設定
            filename = os.path.splitext(os.path.basename(image_path))[0]
            # マスク処理した画像の名前を設定
            image_masked_path = os.path.join(dirs["MASK"], f"{filename}_masked.png")
            # マスク処理を実行 result=action
            result = MaskProcessing(image_path, image_masked_path)

            if result["quit"]:
                print("Processing stopped by user.")
                break

            # 教師データの画像の名前を設定
            # image_corrected_path = os.path.join(dirs["TEACHER"], f"{filename}_corrected.png")
            # データ作成 エラーが出るなら止める
    
            # image_masked_path(マスク処理された画像)を使い,ヒストグラム(CSV)を作成,histpreディレクトリに保存
            CreateHistogram(image_masked_path, dirs["HIST"])
            CreateHistogram5(image_masked_path, dirs["HIST"])

            # filename + ".png"= real_rgb_jsonの時のデータと image_masked_path(マスク処理された画像)を使い,結果はヒストグラム(CSV)を作成,teacherhistディレクトリに保存
            # 教師データの処理を実行、学習に使わないので、ここではコメントアウト
            # TeacherProcessing(filename , dirs["REAL_RGB_JSON"], image_masked_path, image_corrected_path)

            #image_corrected_path(マスク処理された教師データの画像)を使い,ヒストグラム(CSV)を作成,teacherhistディレクトリに保存
            # 教師用のヒストグラムを作成するが、学習には使わないので、ここではコメントアウト
            # CreateHistogram(image_corrected_path, dirs["TEACHER_HIST"])

            # ここで画像を END ディレクトリに移動 本番では使用する
            # 使い終わった画像を END ディレクトリに移動
            # dst_path = os.path.join(dirs["END"], os.path.basename(image_path))
            # shutil.move(image_path, dst_path)
            # print(f"Moved {image_path} → {dst_path}")

        except Exception as e:
            print(f"処理中にエラーが発生しました: {e}")

# 実行
if __name__ == "__main__":
    pretreatment()
