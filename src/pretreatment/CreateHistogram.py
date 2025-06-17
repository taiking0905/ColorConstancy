import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import pandas as pd

from config import move_figure

# ヒストグラムを作成し、表示する関数
def CreateHistogram(image_path, output_path):
    bin_width = 0.02
    num_bins = int(1.0 / bin_width)
    filename = os.path.splitext(os.path.basename(image_path))[0]
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # # 黒以外の領域をマスク（正規化後の値に対応）
    # mask = cv2.inRange(img, (1e-6, 1e-6, 1e-6), (1.0, 1.0, 1.0))
    # valid_mask = mask.flatten() > 0

    # ======================
    # RGB ヒストグラム表示
    # ======================

    display_img = (img * 255).astype('uint8')
    mask = cv2.inRange(display_img, (1, 1, 1), (255, 255, 255))  # 表示には使える
    norm_display = display_img.astype('float32') / 255.0

    fig1 = plt.figure(figsize=(10, 4))
    for i, color in enumerate(('r', 'g', 'b')):
        hist = cv2.calcHist([norm_display], [i], mask, [256], [0, 1])
        plt.plot(hist, color=color, label=f"{color.upper()} channel")
    plt.title(f"RGB Histogram: {filename}")
    plt.xlabel("Intensity")
    plt.ylabel("Pixel Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # ======================
    # 2D ヒストグラム（rg空間）
    # ======================
    sum_rgb = np.sum(img, axis=2, keepdims=True) + 1e-6
    rgb_ratio = img / sum_rgb
    r_ratio = rgb_ratio[:, :, 2]  # R
    g_ratio = rgb_ratio[:, :, 1]  # G

    valid_mask = (sum_rgb[:, :, 0] > 1e-6)
    r_bins = np.clip((r_ratio[valid_mask] / bin_width).astype(int), 0, num_bins - 1)
    g_bins = np.clip((g_ratio[valid_mask] / bin_width).astype(int), 0, num_bins - 1)

    binary_hist_2d = np.zeros((num_bins, num_bins), dtype=np.uint8)
    for r_bin, g_bin in zip(r_bins, g_bins):
        binary_hist_2d[g_bin, r_bin] = 1

    # 2D ヒストグラムを表示
    presence_mask = binary_hist_2d > 0

    fig2 = plt.figure(figsize=(6, 6))
    plt.imshow(presence_mask.T, origin='lower', cmap='gray',
            extent=[0, 1, 0, 1], aspect='auto')
    plt.colorbar(label='Pixel Exists (True/False)')
    plt.xlabel('r = R / (R+G+B)')
    plt.ylabel('g = G / (R+G+B)')
    plt.title(f"2D Hist (rg mask): {filename}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # ======================
    # 共通イベントハンドラで同時に閉じる
    # ======================

    # 左端・右端に配置（モニタサイズに応じて調整）
    move_figure(fig1, 0, 100)        # 左端（x=0, y=100）
    move_figure(fig2, 1200, 100)     # 右寄り（x=1200, y=100）※必要に応じて調整

    # ========= Enterで同時に閉じる =========
    def on_key(event):
        if event.key == 'enter':
            plt.close(fig1)
            plt.close(fig2)

    fig1.canvas.mpl_connect("key_press_event", on_key)
    fig2.canvas.mpl_connect("key_press_event", on_key)

    plt.show(block=False)
    plt.show()

    # presence_mask を flatten して1250次元ベクトルとして保存
    flat_mask = []
    for g in range(50):  # y軸（行）
        for r in range(50):  # x軸（列）
            if (r * 0.02 + g * 0.02) <= 1.0:
                flat_mask.append(presence_mask[g, r])  # row, col の順

    flat_mask = np.array(flat_mask).astype(int)
    pd.DataFrame([flat_mask]).to_csv(
        os.path.join(output_path, f"{filename}.csv"), index=False, header=False
    )


    print(f"Saved 1250-dim histogram for: {filename}")