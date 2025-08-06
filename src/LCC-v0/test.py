import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

from load_dataset import load_dataset
from MLPModel import MLPModel, mse_chromaticity_loss, evaluate
from config import TEST_DIR, REAL_RGB_JSON_PATH, OUTPUT_DIR, DEVICE, SEED, set_seed

def compute_angular_errors(y_pred_all, y_true_all):
    y_pred_norm = y_pred_all / np.linalg.norm(y_pred_all, axis=1, keepdims=True)
    y_true_norm = y_true_all / np.linalg.norm(y_true_all, axis=1, keepdims=True)
    dot_products = np.clip(np.sum(y_pred_norm * y_true_norm, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(dot_products))

def visualize_errors(y_pred_all, y_true_all, angular_errors):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # (1) ヒストグラム
    plt.figure(figsize=(6, 4))
    plt.hist(angular_errors, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel("Angular Error (°)")
    plt.ylabel("Count")
    plt.title("Angular Error Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "angular_error_histogram.png")
    plt.close()

    # (2) 箱ひげ図
    plt.figure(figsize=(4, 6))
    plt.boxplot(angular_errors, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightgreen'),
                medianprops=dict(color='red'))
    plt.ylabel("Angular Error (°)")
    plt.title("Angular Error Boxplot")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "angular_error_boxplot.png")
    plt.close()

    # (3) サンプルごとの角度誤差
    plt.figure(figsize=(8, 4))
    plt.scatter(np.arange(len(angular_errors)), angular_errors, c='orange', alpha=0.6)
    plt.axhline(np.mean(angular_errors), color='red', linestyle='--', label=f"Mean = {np.mean(angular_errors):.2f}°")
    plt.xlabel("Sample Index")
    plt.ylabel("Angular Error (°)")
    plt.title("Angular Error per Sample")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "angular_error_scatter.png")
    plt.close()

def tri_mean(data):
    """Tri-mean = (Q1 + 2*Q2 + Q3) / 4"""
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    return (q1 + 2*q2 + q3) / 4

def best_25_percent(data):
    """最も小さい25%の平均"""
    k = max(1, int(len(data) * 0.25))
    return np.mean(np.sort(data)[:k])

def worst_25_percent(data):
    """最も大きい25%の平均"""
    k = max(1, int(len(data) * 0.25))
    return np.mean(np.sort(data)[-k:])

def main():
    set_seed(SEED)

    # 1. テストデータの読み込み
    X_test_np, y_test_np = load_dataset(TEST_DIR, REAL_RGB_JSON_PATH)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)
    
    # 2. モデル構造の再定義
    model = MLPModel(input_dim=X_test.shape[1], hidden_dim=256, output_dim=2)
    model.load_state_dict(torch.load(OUTPUT_DIR / 'mlp_model.pth', map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 3. 評価実行（クロマティシティMSE）
    test_loss = evaluate(model, test_loader, mse_chromaticity_loss)
    print(f"📊 Test Loss = {test_loss:.4f}")

    # 4. 予測と実際のRGB値の表示（5件だけ）
    print("\n🎨 Prediction vs Actual (first 5 samples):")
    num_samples = min(5, len(X_test))
    with torch.no_grad():
        for i in range(num_samples):
            x = X_test[i].unsqueeze(0).to(DEVICE)
            pred = model(x)[0]

            r_pred, g_pred = pred[0].item(), pred[1].item()
            b_pred = max(0, 1 - (r_pred + g_pred))

            r_true, g_true, b_true = y_test[i].cpu().numpy()
            print(f"{i+1}: Pred (r, g, b): ({r_pred:.4f}, {g_pred:.4f}, {b_pred:.4f}) | True (r, g, b): ({r_true:.4f}, {g_true:.4f}, {b_true:.4f})")

    # 5. 全データで予測・評価
    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(DEVICE)
            preds = model(x_batch).cpu().numpy()
            b_preds = np.maximum(0, 1 - preds[:, 0] - preds[:, 1])
            preds_3d = np.column_stack([preds, b_preds])

            y_pred_list.append(preds_3d)
            y_true_list.append(y_batch.numpy())

    y_pred_all = np.vstack(y_pred_list)
    y_true_all = np.vstack(y_true_list)

    angular_errors = compute_angular_errors(y_pred_all, y_true_all)
    visualize_errors(y_pred_all, y_true_all, angular_errors)

    print("\n📏 Angular Error Stats (°):")
    print(f"Method\t\tValue")
    print(f"Mean\t\t{np.mean(angular_errors):.4f}")
    print(f"Median\t\t{np.median(angular_errors):.4f}")
    print(f"Tri-m.\t\t{tri_mean(angular_errors):.4f}")
    print(f"B-25\t\t{best_25_percent(angular_errors):.4f}")
    print(f"W-25\t\t{worst_25_percent(angular_errors):.4f}")
    print(f"95-P\t\t{np.percentile(angular_errors, 95):.4f}")
    print(f"99-P\t\t{np.percentile(angular_errors, 99):.4f}")
    print(f"Max\t\t{np.max(angular_errors):.4f}")

if __name__ == "__main__":
    main()
