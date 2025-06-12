import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import glob
import pandas as pd
import json
import shutil 
matplotlib.use('TkAgg') 
from pathlib import Path

# ==== ディレクトリ設定 ====

def setup_directories():
    base = Path("..")
    dirs = {
        "input": base / "datapre",
        "mask": base / "maskpre",
        "hist": base / "histpre",
        "teacher": base / "teacher",
        "teacher_hist": base / "teacherhist",
        "real_rgb_json": base / "real_rgb.json",
        "endpic": base / "endpic"
    }

    # ディレクトリがなければ作成（JSONはスキップ）
    for key, path in dirs.items():
        if key == "real_rgb_json":
            continue
        path.mkdir(parents=True, exist_ok=True)

    # JSONファイルの存在チェック
    if not dirs["real_rgb_json"].exists():
        print("⚠ real_rgb.json が存在しません。先に作成してください。")
    
    return dirs



# グローバルフラグ
action = {"next": False, "quit": False} # enterで次へ進むためのフラグ,qでプログラムを終了するためのフラグ

# ウィンドウを左端・右端に配置する関数
def move_figure(fig, x, y):
    backend = plt.get_backend()
    if backend == 'TkAgg':
        fig.canvas.manager.window.wm_geometry(f"+{x}+{y}")
    else:
        print(f"ウィンドウ移動はこのバックエンドでは未対応: {backend}")

# マスク処理を行う関数
def MaskProcessing(image_path, output_path):
    global action
    coords = [] # クリックした座標を保存するリスト
    action = {"next": False, "quit": False}  # 初期化

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img_scaled = (img / 16).astype("uint8")

    # ガンマ補正を適用する関数(表示専用)
    def apply_gamma_correction(image, gamma=2.2):
        img_float = image / 255.0
        img_gamma = np.power(img_float, 1.0 / gamma)
        return (img_gamma * 255).astype("uint8")

    img_display = apply_gamma_correction(img_scaled)
    img_rgb_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

    # マスク処理で、クリックした領域を黒く塗りつぶす関数
    def mask_region_and_save():
        mask = np.ones_like(img_scaled, dtype=np.uint8) * 255
        pts = np.array([coords], dtype=np.int32)
        cv2.fillPoly(mask, pts, (0, 0, 0))
        masked_img = cv2.bitwise_and(img_scaled, mask)
        cv2.imwrite(output_path, masked_img)
        print(f"Saved masked image to: {output_path}")
        ax.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
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
        global action
        if event.key == 'r':
            coords.clear()
            ax.clear()
            ax.imshow(img_rgb_display)
            ax.set_title("Click 4 points (Press 'r' to reset)")
            fig.canvas.draw()
        elif event.key == 'enter':
            action["next"] = True
            plt.close()
        elif event.key == 'q':
            action["quit"] = True
            plt.close()

    fig, ax = plt.subplots(figsize=(10, 9)) 
    move_figure(fig, 0, 0) 
    ax.imshow(img_rgb_display)
    ax.set_title("Click 4 points (Press 'r' to reset)")
    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)
    plt.show()

    return action

# ヒストグラムを作成し、表示する関数
def CreateHistogram(image_path, output_path):

    filename = os.path.splitext(os.path.basename(image_path))[0]
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # 黒以外の領域をマスク0を除外する
    mask = cv2.inRange(img, (1, 1, 1), (255, 255, 255))
    valid_mask = mask.flatten() > 0

    # ======================
    # RGB ヒストグラム表示
    # ======================
    norm_img = img.astype('float32') / 255.0
    fig1 = plt.figure(figsize=(10, 4))
    for i, color in enumerate(('r', 'g', 'b')):
        hist = cv2.calcHist([norm_img], [i], mask, [256], [0, 1])
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
    R = img[:, :, 0].astype('float32')
    G = img[:, :, 1].astype('float32')
    B = img[:, :, 2].astype('float32')
    total = R + G + B + 1e-6
    r = R / total
    g = G / total
    r_flat = r.flatten()[valid_mask]
    g_flat = g.flatten()[valid_mask]

    # 0.02区切りでヒストグラムを作成
    hist2d, _, _ = np.histogram2d(r_flat, g_flat, bins=[50, 50], range=[[0, 1], [0, 1]])
    # ヒストグラムをbooleanマスクに変換
    presence_mask = hist2d > 0

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
    for i in range(50):
        for j in range(50):
            r_val = i * 0.02
            g_val = j * 0.02
            if r_val + g_val <= 1.0:
                flat_mask.append(presence_mask[i, j])

    # numpy配列に変換し、1行のCSVとして保存
    flat_mask = np.array(flat_mask).astype(int)
    pd.DataFrame([flat_mask]).to_csv(
        os.path.join(output_path, f"{filename}.csv"), index=False, header=False
    )

    print(f"Saved 1250-dim histogram for: {filename}")

    # テスト用正規化できている
    #print(f"{filename}: max(r + g) = {np.max(r_flat + g_flat)}")

# 教師データを処理する関数
def TeacherProcessing(filename, real_rgb_json, image_path, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    
    # 教師データ読み込み
    with open(real_rgb_json, "r") as f:
        real_rgb_data = json.load(f)
    
    real_rgb = None
    # 対応するreal_rgbを探す
    for item in real_rgb_data:
        if item["filename"] == filename:
            real_rgb = np.array(item["real_rgb"], dtype=np.float32)
            break
    
    if real_rgb is None:
        raise ValueError(f"{filename} の real_rgb が見つかりません。")
    
    # スケーリング係数計算
    scaling_factors = real_rgb / np.mean(real_rgb)
    
    # スケーリング係数を適用
    img_corrected = img / scaling_factors  # ブロードキャストされる
    
    # クリップして0〜1に正規化 (ここは元画像のスケールに依存)
    img_corrected = np.clip(img_corrected, 0, 255)
    
    # uint8に変換して保存
    img_to_save = img_corrected.astype(np.uint8)
    cv2.imwrite(output_path, img_to_save)
    print(f"Corrected image saved to: {output_path}")
    
    # 表示（matplotlib）→ Enter 押しで閉じる
    fig, ax = plt.subplots(figsize=(12, 9)) 
    move_figure(fig, 0, 0) 
    plt.imshow(cv2.cvtColor(img_to_save, cv2.COLOR_BGR2RGB))
    plt.title("Corrected Image - Press Enter to close")
    plt.axis("off")

    def on_key(event):
        if event.key == 'enter':
            plt.close()

    fig = plt.gcf()
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
       
def pretreatment():
    # ディレクトリ設定
    dirs = setup_directories()

    # png画像のパスを取得
    image_paths = sorted(glob.glob(os.path.join(dirs["input"], "*.png")))

    
    for image_path in image_paths:
        try:
            # 画像ファイル名から拡張子を除去して、マスクと教師データのパスを設定
            filename = os.path.splitext(os.path.basename(image_path))[0]
            # マスク処理した画像の名前を設定
            image_masked_path = os.path.join(dirs["mask"], f"{filename}_masked.png")
            # マスク処理を実行 result=action
            result = MaskProcessing(image_path, image_masked_path)

            if result["quit"]:
                print("Processing stopped by user.")
                break

            # 教師データの画像の名前を設定
            image_corrected_path = os.path.join(dirs["teacher"], f"{filename}_corrected.png")
            # データ作成 エラーが出るなら止める
    
            # image_masked_path(マスク処理された画像)を使い,ヒストグラム(CSV)を作成,histpreディレクトリに保存
            CreateHistogram(image_masked_path, dirs["hist"])

            # filename + ".png"= real_rgb_jsonの時のデータと image_masked_path(マスク処理された画像)を使い,結果はヒストグラム(CSV)を作成,teacherhistディレクトリに保存
            TeacherProcessing(filename , dirs["real_rgb_json"], image_masked_path, image_corrected_path)

            #image_corrected_path(マスク処理された教師データの画像)を使い,ヒストグラム(CSV)を作成,teacherhistディレクトリに保存
            CreateHistogram(image_corrected_path, dirs["teacher_hist"])        

            # ここで画像を endpic_dir に移動 本番では使用する
            # 使い終わった画像を endpic ディレクトリに移動
            # dst_path = os.path.join(dirs[endpic], os.path.basename(image_path))
            # shutil.move(image_path, dst_path)
            # print(f"Moved {image_path} → {dst_path}")

        except Exception as e:
            print(f"処理中にエラーが発生しました: {e}")

# 実行
if __name__ == "__main__":
    pretreatment()
