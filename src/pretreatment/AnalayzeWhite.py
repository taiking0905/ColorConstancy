import matplotlib.pyplot as plt
import cv2
import numpy as np

from config import move_figure

def analyze_white_patch(patch_path, gamma=2.2, threshold=0.05):
    # 画像の読み込み（補正前前提）
    img = cv2.imread(patch_path).astype(np.float32)

    # 表示用（γ補正 + 拡大）
    img_normalized = img / 255.0
    display_img = np.power(img_normalized, 1.0 / gamma)
    display_img = cv2.resize(display_img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
   
    coords = []

    def onclick(event):
        if event.xdata and event.ydata and len(coords) < 4:
            x, y = int(event.xdata), int(event.ydata)
            coords.append((x, y))
            ax.plot(x, y, 'go')
            ax.text(x+5, y-5, f"{len(coords)}", color="green", fontsize=12)
            fig.canvas.draw()
            if len(coords) == 4:
                plt.close()

    fig, ax = plt.subplots(figsize=(8, 8))
    move_figure(fig, 0, 20)
    ax.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
    ax.set_title("Click 4 points for white patch region")
    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    if len(coords) < 4:
        print("❌ 4点選択されませんでした。中止します。")
        return None

    # 座標をリサイズ前に戻す
    coords = np.array(coords) / 4.0
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [coords.astype(np.int32)], 1)

    # 選択領域のピクセルを取得
    region_pixels = img[mask == 1].reshape(-1, 3)

    # 色のユニーク集計
    colors, counts = np.unique((region_pixels * 255).astype(np.uint8), axis=0, return_counts=True)
    total = counts.sum()
    ratios = counts / total

    # フィルタ処理（影響力5%以上）
    mask_valid = ratios >= threshold
    filtered_colors = colors[mask_valid]
    filtered_counts = counts[mask_valid]

    if len(filtered_colors) == 0:
        print("❌ 有効な色が存在しません（5%以上の色なし）")
        return None

    # 平均色を重み付きで計算
    mean_color = (filtered_colors * filtered_counts[:, None]).sum(axis=0) / filtered_counts.sum()
    print("📊 平均色 (R,G,B):", mean_color.astype(int))

    # 色詳細の表示
    for i, (col, cnt) in enumerate(zip(filtered_colors, filtered_counts)):
        print(f"{i+1}. {col} : {cnt} pixels ({cnt / total:.2%})")

    return mean_color
