import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import pandas as pd

from config import move_figure

bin_width = 0.02
num_bins = int(1.0 / bin_width)

def load_and_normalize_image(image_path):
    filename = os.path.splitext(os.path.basename(image_path))[0]
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    black_mask = np.any(img > 0, axis=2)
    sum_rgb = np.sum(img, axis=2, keepdims=True) + 1e-6
    rgb_ratio = img / sum_rgb
    return filename, img, sum_rgb, rgb_ratio, black_mask


# ヒストグラムを作成し表示する関数
def CreateHistogram(image_path, output_path):
    
    filename, img, sum_rgb, rgb_ratio, black_mask=load_and_normalize_image(image_path)


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
    ax1 = fig1.add_subplot(1, 1, 1)
    for i, color in enumerate(('r', 'g', 'b')):
        hist = cv2.calcHist([norm_display], [i], mask, [256], [0, 1])
        ax1.plot(hist, color=color, label=f"{color.upper()} channel")
    ax1.set_title(f"RGB Histogram: {filename}")
    ax1.set_xlabel("Intensity")
    ax1.set_ylabel("Pixel Count")
    ax1.legend()
    ax1.grid(True)
    fig1.tight_layout()


    # ======================
    # 2D ヒストグラム（rg空間）
    # ======================
    r_ratio = rgb_ratio[:, :, 2]  # R
    g_ratio = rgb_ratio[:, :, 1]  # G
    valid_mask = black_mask & (sum_rgb[:, :, 0] > 1e-6)
    r_bins = np.clip((r_ratio[valid_mask] / bin_width).astype(int), 0, num_bins - 1)
    g_bins = np.clip((g_ratio[valid_mask] / bin_width).astype(int), 0, num_bins - 1)

    binary_hist_2d = np.zeros((num_bins, num_bins), dtype=np.uint8)
    for r_bin, g_bin in zip(r_bins, g_bins):
        binary_hist_2d[g_bin, r_bin] = 1

    # 2D ヒストグラムを表示
    presence_mask = binary_hist_2d > 0

    fig2 = plt.figure(figsize=(6, 6))
    ax2 = fig2.add_subplot(1, 1, 1)
    im2 = ax2.imshow(presence_mask.T, origin='lower', cmap='gray',
                    extent=[0, 1, 0, 1], aspect='auto')
    ax2.set_title(f"2D Hist (rg mask): {filename}")
    ax2.set_xlabel('r = R / (R+G+B)')
    ax2.set_ylabel('g = G / (R+G+B)')
    ax2.grid(True, alpha=0.3)
    fig2.colorbar(im2, ax=ax2, label='Pixel Exists (True/False)')

    # ======================
    # 共通イベントハンドラで同時に閉じる
    # ======================

    # 左端・右端に配置（モニタサイズに応じて調整）
    move_figure(fig1, 0, 20)        # 左端
    move_figure(fig2, 60, 20)     # 右寄り

    # ========= Enterで同時に閉じる =========
    def on_key(event):
        if event.key == 'enter':
            plt.close(fig1)
            plt.close(fig2)

    fig1.canvas.mpl_connect("key_press_event", on_key)
    fig2.canvas.mpl_connect("key_press_event", on_key)

    plt.show(block=True )

    # presence_mask を flatten して1250次元ベクトルとして保存
    flat_mask = []
    for g in range(50):  # y軸（行）
        for r in range(50):  # x軸（列）
            if (r * bin_width + g * bin_width) <= 1.0:
                flat_mask.append(presence_mask[g, r])  # row, col の順

    flat_mask = np.array(flat_mask).astype(int)
    pd.DataFrame([flat_mask]).to_csv(
        os.path.join(output_path, f"{filename}.csv"), index=False, header=False
    )


    print(f"Saved 1250-dim histogram for: {filename}")

def CreateHistogram_rg_gb(image_path, output_path):
    filename, _,sum_rgb, rgb_ratio, black_mask=load_and_normalize_image(image_path)

    # ==============================
    # 2D ヒストグラム（rg空間 &gb空間）
    # ==============================
    r_ratio = rgb_ratio[:, :, 2]  # R
    g_ratio = rgb_ratio[:, :, 1]  # G
    b_ratio = rgb_ratio[:, :, 0]  # B
    valid_mask = black_mask & (sum_rgb[:, :, 0] > 1e-6)
    r_bins = np.clip((r_ratio[valid_mask] / bin_width).astype(int), 0, num_bins - 1)
    g_bins = np.clip((g_ratio[valid_mask] / bin_width).astype(int), 0, num_bins - 1)
    b_bins = np.clip((b_ratio[valid_mask] / bin_width).astype(int), 0, num_bins - 1)

    hist_rg_2d = np.zeros((num_bins, num_bins), dtype=np.uint32)
    hist_gb_2d = np.zeros((num_bins, num_bins), dtype=np.uint32)
    

    for r_bin, g_bin in zip(r_bins, g_bins):
        hist_rg_2d[g_bin, r_bin] += 1

    for g_bin, b_bin in zip(g_bins, b_bins):
        hist_gb_2d[b_bin, g_bin] += 1

    if hist_rg_2d.max() > 0:
        presence_rg = (hist_rg_2d / hist_rg_2d.max()).astype(np.float32)
    else:
        presence_rg = hist_rg_2d.astype(np.float32)

    if hist_gb_2d.max() > 0:
        presence_gb = (hist_gb_2d / hist_gb_2d.max()).astype(np.float32)
    else:
        presence_gb = hist_gb_2d.astype(np.float32)


    upsampled_rg = cv2.resize(presence_rg, (224, 224), interpolation=cv2.INTER_LINEAR)
    upsampled_gb = cv2.resize(presence_gb, (224, 224), interpolation=cv2.INTER_LINEAR)

    rg_hist_8bit = (upsampled_rg * 255).astype(np.uint8)
    gb_hist_8bit = (upsampled_gb * 255).astype(np.uint8)

    # 180度回転
    upsampled_gb_rotated = np.rot90(gb_hist_8bit, 2)

    combined = np.zeros((224, 224), dtype=np.float32)

    for y in range(224):
        for x in range(224):
            if x + y <= 224:  # 境界を rg 側に含める
                combined[y, x] = rg_hist_8bit[y, x]
            else:
                combined[y, x] = upsampled_gb_rotated[y, x]

    stacked = np.stack([combined], axis = 0)  # (1, 224, 224)

    np.save(os.path.join(output_path, f"{filename}.npy"), stacked)
    print(f"Saved upsampled 224x224x2 histogram to: {filename}.npy")


    fig1 = plt.figure(figsize=(6, 6))
    ax1 = fig1.add_subplot(1, 1, 1)
    from matplotlib.colors import LogNorm
    im1 = ax1.imshow(hist_rg_2d.T, origin='lower', cmap='viridis',
                 extent=[0, 1, 0, 1], aspect='auto',
                 norm=LogNorm(vmin=1, vmax=255))

    ax1.set_title(f"2D Hist (rg count): {filename}")
    ax1.set_xlabel('r = R / (R+G+B)')
    ax1.set_ylabel('g = G / (R+G+B)')
    ax1.grid(True, alpha=0.3)
    fig1.colorbar(im1, ax=ax1, label='Pixel Exists')

    fig2 = plt.figure(figsize=(6, 6))
    ax2 = fig2.add_subplot(1, 1, 1)
    im2 = ax2.imshow(hist_gb_2d.T, origin='lower', cmap='viridis',
                 extent=[0, 1, 0, 1], aspect='auto',
                 norm=LogNorm(vmin=1, vmax=255))
    ax2.set_title(f"2D Hist (gb count): {filename}")
    ax2.set_xlabel('g = G / (R+G+B)')
    ax2.set_ylabel('b = B / (R+G+B)')
    ax2.grid(True, alpha=0.3)
    fig2.colorbar(im2, ax=ax2, label='Pixel Count (log scale)')

    fig3 = plt.figure(figsize=(6, 6))
    ax3 = fig3.add_subplot(1, 1, 1)
    im3 = ax3.imshow(combined.T, origin='lower', cmap='viridis',
                 extent=[0, 1, 0, 1], aspect='auto',
                 norm=LogNorm(vmin=1, vmax=255))
    ax3.set_title(f"RG & GB Combined Histogram: {filename}")
    ax3.grid(True, alpha=0.3)
    fig3.colorbar(im3, ax=ax3, label='Presence Value (log scale)')


    # ======================
    # 共通イベントハンドラで同時に閉じる
    # ======================

    # 左端・右端に配置（モニタサイズに応じて調整）
    move_figure(fig1, 0, 20)        # 左端（x=0, y=100）
    move_figure(fig2, 30, 20)     # 右寄り（x=1200, y=100）※必要に応じて調整
    move_figure(fig3, 60, 20)

    # ========= Enterで同時に閉じる =========
    def on_key(event):
        if event.key == 'enter':
            plt.close(fig1)
            plt.close(fig2)
            plt.close(fig3)

    fig1.canvas.mpl_connect("key_press_event", on_key)
    fig2.canvas.mpl_connect("key_press_event", on_key)
    fig3.canvas.mpl_connect("key_press_event", on_key)

    plt.show(block=True)

    print(f"Saved 1250-dim histogram for: {filename}")