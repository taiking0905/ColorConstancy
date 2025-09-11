import cv2
import numpy as np

BLACK_LEVEL = 528
WHITE_LEVEL = 4095

def to_8bit_gamma(img, gamma=2.2):
    # 正規化
    img_norm = np.clip((img - BLACK_LEVEL) / (WHITE_LEVEL - BLACK_LEVEL), 0, WHITE_LEVEL)

    # ガンマ補正
    img_gamma = np.power(img_norm, 1/gamma)

    return (img_gamma * 255).astype(np.uint8)

# 画像読み込み
img_path = 'C:/Users/taiki/Desktop/test/IMG_4017.png'
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

print("読み込み確認:", img.shape, img.dtype, img.min(), img.max())

# ガンマ変換
img_display = to_8bit_gamma(img)

cv2.imshow("Gamma corrected", cv2.resize(img_display, (800, 600)))
cv2.waitKey(0)
cv2.destroyAllWindows()
