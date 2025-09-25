import time
import rawpy
import numpy as np
import cv2
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv

load_dotenv()
OneDrive_DATA_PATH = os.getenv("OneDrive_DATA_PATH")
OneDrive_PNG_PATH = os.getenv("OneDrive_PNG_PATH")

class DNGHandler(FileSystemEventHandler):
    def process_dng(self, dng_path):
        try:
            with rawpy.imread(dng_path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=False,
                    no_auto_bright=True,
                    no_auto_scale=True,
                    output_bps=16,
                    gamma=(1, 1),
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                    output_color=rawpy.ColorSpace.raw
                )
                filename = os.path.splitext(os.path.basename(dng_path))[0] + ".png"
                save_path = os.path.join(OneDrive_PNG_PATH, filename)
                success = cv2.imwrite(save_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                if success:
                    print(f"保存成功: {save_path}")
                else:
                    print(f"保存失敗: {save_path} （フォルダ存在しない可能性あり）")
                
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