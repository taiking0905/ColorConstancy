import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from ResNetModel import ResNetModel
from config import DEVICE, OUTPUT_DIR


def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def estimate_illumination(model, npy_feature):
    x = torch.tensor(npy_feature, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(x)[0].cpu().numpy()
    return normalize_vector(pred)

def correct_image_color(image_rgb, illumination_vector):
    illumination_vector = normalize_vector(illumination_vector)
    gain = np.mean(illumination_vector) / illumination_vector
    corrected = image_rgb.astype(np.float32) / 255.0
    corrected *= gain
    corrected = np.clip(corrected, 0, 1)
    return (corrected * 255).astype(np.uint8)

def apply_additional_corrections(image_rgb, wb_multipliers=None, gamma=2.2):
    """
    ホワイトバランス＋ガンマ補正を行う（image_rgbは0~255のuint8）
    wb_multipliers: [R, G, B]の乗算係数（Noneなら適用しない）
    """
    img = image_rgb.astype(np.float32) / 255.0
    if wb_multipliers is not None:
        for i in range(3):
            img[..., i] *= wb_multipliers[i]
    img = np.clip(img, 0, 1)
    img = np.power(img, 1.0 / gamma)
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def main(png_path: Path, npy_path: Path):
    # モデルロード
    model = ResNetModel().to(DEVICE)
    state_dict = torch.load(OUTPUT_DIR / 'resnet_model.pth')
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    # 画像読み込み（BGR->RGB）
    image_rgb= cv2.imread(str(png_path))
    # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 特徴量読み込み
    npy_feature = np.load(npy_path)

    # 照明推定
    illumination = estimate_illumination(model, npy_feature)
    print(f"Estimated illumination: {illumination}")

    # 照明推定だけの補正画像（簡易）
    corrected_rgb = correct_image_color(image_rgb, illumination)

    # 追加補正（例：AsShotNeutralの逆数を利用してWB補正＋ガンマ補正）
    # 実際は自分の環境で取得したAsShotNeutralを入れてください
    as_shot_neutral = np.array([2.562066,0.9945468,1.6436796], dtype=np.float32)
    wb_multipliers = 1.0 / as_shot_neutral

    display_rgb = apply_additional_corrections(image_rgb, wb_multipliers=wb_multipliers, gamma=2.8)
    corrected_rgb_plus = apply_additional_corrections(corrected_rgb, wb_multipliers=wb_multipliers, gamma=2.8)


    # 保存（個別に2枚）
    save_path_output = OUTPUT_DIR / f"test_{png_path.name}"
    save_path_illum = OUTPUT_DIR / f"illum_corrected_{png_path.name}"
    save_path_plus = OUTPUT_DIR / f"wb_gamma_corrected_{png_path.name}"
    cv2.imwrite(str(save_path_output), cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(save_path_illum), cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(save_path_plus), cv2.cvtColor(corrected_rgb_plus, cv2.COLOR_RGB2BGR))
    print(f"✅ 照明推定補正画像を保存しました: {save_path_illum}")
    print(f"✅ WB+ガンマ補正画像を保存しました: {save_path_plus}")

    # 表示
    plt.figure(figsize=(12, 6))
    plt.imshow(corrected_rgb_plus)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    png_file = Path(OUTPUT_DIR / "output.png")
    npy_file = Path(OUTPUT_DIR / "output.npy")
    main(png_file, npy_file)
