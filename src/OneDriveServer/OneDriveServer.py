import time
import rawpy
import numpy as np
import cv2
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv

# OneDriveフォルダー設定
load_dotenv()
OneDrive_DATA_PATH = os.getenv("OneDrive_DATA_PATH")
OneDrive_RAW_PNG_PATH = os.getenv("OneDrive_RAW_PNG_PATH")
OneDrive_GANMA_PNG_PATH = os.getenv("OneDrive_GANMA_PNG_PATH")
Onedrive_TRASH_BOX_PATH = os.getenv("Onedrive_TRASH_BOX_PATH")

# カメラ設定（black/whiteレベル）
BLACK_LEVEL = 528
WHITE_LEVEL = 4095

def to_8bit_gamma(img, gamma=1):
    """
    12bitまたは16bit画像を8bitに変換して、ガンマ補正も適用（表示用）
    """
    # 正規化（0〜1）
    img = np.clip((img)/ (WHITE_LEVEL - BLACK_LEVEL), 0, 1)

    # ガンマ補正（sRGB風）
    img_gamma = np.power(img, 1 / gamma)

    # 8bit化
    return (img_gamma * 255).astype(np.uint8)

class DNGHandler(FileSystemEventHandler):
    def process_dng(self, dng_path):
        try:
            with rawpy.imread(dng_path) as raw:
                rgb_raw = raw.postprocess(
                    use_camera_wb=False,
                    no_auto_bright=True,
                    no_auto_scale=True,
                    output_bps=16,
                    gamma=(1, 1),
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                    output_color=rawpy.ColorSpace.raw
                )
                filename_raw = os.path.splitext(os.path.basename(dng_path))[0] + ".png"
                save_path_raw = os.path.join(OneDrive_RAW_PNG_PATH, filename_raw)
                success_raw = cv2.imwrite(save_path_raw, cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR))
                if success_raw:
                    print(f"保存成功: {save_path_raw}")
                else:
                    print(f"保存失敗: {save_path_raw} （フォルダ存在しない可能性あり）")
                
                # ② 8bit化＋ガンマ補正画像を保存
                rgb_gamma = to_8bit_gamma(rgb_raw, gamma=1)
                filename_gamma = os.path.splitext(os.path.basename(dng_path))[0] + "_gamma.jpg"
                save_path_gamma = os.path.join(OneDrive_GANMA_PNG_PATH, filename_gamma)
                success_gamma = cv2.imwrite(save_path_gamma, cv2.cvtColor(rgb_gamma, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                if success_gamma:
                    print(f"保存成功 (Gamma JPEG): {save_path_gamma}")
                else:
                    print(f"保存失敗 (Gamma JPEG): {save_path_gamma}")

        except Exception as e:
            print(f"処理中にエラーが発生しました ({dng_path}): {e}")

    def on_created(self, event):
        if event.src_path.lower().endswith(".dng"):
            print(f"新規ファイル検知: {event.src_path}")
            time.sleep(5)
            self.process_dng(event.src_path)

if __name__ == "__main__":
    event_handler = DNGHandler()
    observer = Observer()
    observer.schedule(event_handler, OneDrive_DATA_PATH, recursive=False)
    observer.start()
    print(f"{OneDrive_DATA_PATH} を監視開始...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()