import matplotlib.pyplot as plt
import cv2
import numpy as np

from config import BLACK_LEVEL, WHITE_LEVEL, move_figure

# マスク処理を行う関数
def MaskProcessing(image_path, output_path, checker_path):
    coords = [] # クリックした座標を保存するリスト
    ACTION = {"next": False, "quit": False}  # 初期化

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    # black level 補正と正規化
    img = np.clip(img - BLACK_LEVEL, 0, None)
    img /= (WHITE_LEVEL - BLACK_LEVEL)

    # ガンマ補正を適用する関数(表示専用)
    def apply_gamma_correction( gamma=2.2):
        img_gamma = np.power(img, 1.0 / gamma)
        return (img_gamma * 255).astype("uint8")

    img_display = apply_gamma_correction()
    img_rgb_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

    # マスク処理で、クリックした領域を黒く塗りつぶす関数
    def mask_region_and_save():
        mask = np.ones_like(img, dtype=np.float32)
        pts = np.array([coords], dtype=np.int32)
        cv2.fillPoly(mask, pts, (0.0, 0.0, 0.0))
        masked_img = img * mask

        # 表示用マスク（uint8）
        mask_display = np.ones_like(img_display, dtype=np.uint8) * 255
        cv2.fillPoly(mask_display, pts, (0, 0, 0))
        masked_display = cv2.bitwise_and(img_display, mask_display)

        # マスク画像保存
        cv2.imwrite(output_path, (masked_img * 255).astype(np.uint8))
        print(f"Saved masked image to: {output_path}")

        # 🔽 ここから追加処理 🔽
        # カラーチェッカー領域だけを切り出して保存・表示
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        cropped_checker = img[y:y+h, x:x+w]
        cropped_checker_uint8 = (cropped_checker * 255).astype(np.uint8)
    
        cv2.imwrite(checker_path, cropped_checker_uint8)
        print(f"Saved cropped checker region to: {checker_path}")

        # 表示にも追加
        ax.imshow(cv2.cvtColor(masked_display, cv2.COLOR_BGR2RGB))
        ax.set_title("Press Enter to continue or q to quit")
        fig.canvas.draw()

    # 4点クリックしたらマスク処理を行うためのイベントハンドラ
    def onclick(event):
        if event.xdata and event.ydata and len(coords) < 4:
            x, y = int(event.xdata), int(event.ydata)
            coords.append((x, y))
            ax.plot(x, y, 'ro')
            ax.text(x+5, y-5, f"{len(coords)}", color="red", fontsize=12)
            fig.canvas.draw()
            if len(coords) == 4:
                mask_region_and_save()

    # rキーでリセット、Enterキーで次へ進む、qキーで終了するためのイベントハンドラ
    def onkey(event):
        if event.key == 'r':
            coords.clear()
            ax.clear()
            ax.imshow(img_rgb_display)
            ax.set_title("Click 4 points (Press 'r' to reset)")
            fig.canvas.draw()
        elif event.key == 'enter':
            ACTION["next"] = True
            plt.close()
        elif event.key == 'q':
            ACTION["quit"] = True
            plt.close()

    fig, ax = plt.subplots(figsize=(10, 9)) 
    move_figure(fig, 0, 0) 
    ax.imshow(img_rgb_display)
    ax.set_title("Click 4 points (Press 'r' to reset)")
    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)
    plt.show()

    return ACTION
