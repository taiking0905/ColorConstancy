import torch
import numpy as np

def compute_angular_errors(y_pred_all, y_true_all):
    """
    予測RGBベクトルと真RGBベクトルの間の角度誤差を計算する (PyTorch版)
    """
    # NumPy配列をPyTorch Tensorに変換
    y_pred_all = torch.tensor(y_pred_all, dtype=torch.float32)
    y_true_all = torch.tensor(y_true_all, dtype=torch.float32)

    # L2ノルムで各ベクトルを正規化
    y_pred_norm = y_pred_all / torch.linalg.norm(y_pred_all, dim=1, keepdim=True)
    y_true_norm = y_true_all / torch.linalg.norm(y_true_all, dim=1, keepdim=True)

    # 正規化されたベクトルの内積を計算し、-1.0から1.0の範囲にクリップ
    dot_products = torch.clip(torch.sum(y_pred_norm * y_true_norm, dim=1), -1.0, 1.0)

    # 角度をラジアンで計算し、度数に変換
    return torch.rad2deg(torch.arccos(dot_products))

# 予測RGBベクトル
y_pred_all = [
    [0.2555, 0.4575, 0.2870],
    [0.2176, 0.4551, 0.3273],
    [0.4066, 0.4554, 0.1380]
]

# 真RGBベクトル
y_true_all = [
    [0.1925, 0.4448, 0.3628],
    [0.1971, 0.4462, 0.3567],
    [0.4333, 0.4602, 0.1065]
]

# 角度誤差の計算
angular_errors = compute_angular_errors(y_pred_all, y_true_all)

# 結果の表示
for i, error in enumerate(angular_errors):
    print(f"Sample {i+1}: Angular Error = {error.item():.3f}°")

print(f"\nMean Angular Error: {torch.mean(angular_errors).item():.3f}°")