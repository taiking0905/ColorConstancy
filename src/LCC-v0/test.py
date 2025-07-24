import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

from load_dataset import load_dataset
from MLPModel import MLPModel, mse_chromaticity_loss, evaluate
from config import TEST_DIR,REAL_RGB_JSON_PATH,OUTPUT_DIR, DEVICE, SEED , set_seed

def compute_angular_errors(y_pred_all, y_true_all):
    # 正規化（念のため）
    y_pred_norm = y_pred_all / np.linalg.norm(y_pred_all, axis=1, keepdims=True)
    y_true_norm = y_true_all / np.linalg.norm(y_true_all, axis=1, keepdims=True)

    # cosθ の内積 → θ = arccos(dot)
    dot_products = np.clip(np.sum(y_pred_norm * y_true_norm, axis=1), -1.0, 1.0)
    angles_rad = np.arccos(dot_products)
    angles_deg = np.degrees(angles_rad)
    return angles_deg

def main():
    set_seed(SEED)
    # 1. テストデータの読み込み
    X_test_df, y_test_df = load_dataset(TEST_DIR, REAL_RGB_JSON_PATH)
    X_test = torch.tensor(X_test_df.values, dtype=torch.float32)
    y_test = torch.tensor(y_test_df[["r_ratio", "g_ratio", "b_ratio"]].values, dtype=torch.float32)

    # 2. モデル構造の再定義（構造は学習と同じに！）
    model = MLPModel(input_dim=X_test.shape[1], hidden_dim=256, output_dim=2)
    model.load_state_dict(torch.load(OUTPUT_DIR / 'mlp_model.pth'))  # 学習済みモデルの読み込み
    model.to(DEVICE)  
    model.eval()

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 3. 評価実行（クロマティシティMSE）
    test_loss = evaluate(model, test_loader, mse_chromaticity_loss)
    print(f"📊 Test Loss = {test_loss:.4f}")

    # 4. 予測と実際のRGB値の表示（5件だけ例として）
    print("\n🎨 Prediction vs Actual (first 5 samples):")
    
    num_samples = min(5, len(X_test))  # データ数に合わせる
    with torch.no_grad():
        for i in range(num_samples):
            x = X_test[i].unsqueeze(0).to(DEVICE)  # shape: (1, input_dim)
            pred = model(x)[0]          # クロマティシティ座標 (r, g)

            # モデル出力（r_pred, g_pred）
            r_pred, g_pred = pred[0].item(), pred[1].item()
            b_pred = max(0, 1 - (r_pred + g_pred))
            r_true, g_true, b_true = y_test[i].numpy()
            print(f"{i+1}: Pred (r, g, b): ({r_pred:.4f}, {g_pred:.4f}, {b_pred:.4f}) | True (r, g, b): ({r_true:.4f}, {g_true:.4f}, {b_true:.4f})")

    # 5. 可視化・評価
    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(DEVICE)
            preds = model(x_batch).cpu().numpy()  # shape: (batch, 2)
            # b成分を補完して3次元に
            b_preds = np.maximum(0, 1 - preds[:, 0] - preds[:, 1])
            preds_3d = np.column_stack([preds, b_preds])
            y_pred_list.append(preds_3d)
            y_true_list.append(y_batch.numpy())

    y_pred_all = np.vstack(y_pred_list)  # shape: (N, 3)
    y_true_all = np.vstack(y_true_list)  # shape: (N, 3)

    # ✅ 角度誤差の計算
    angular_errors = compute_angular_errors(y_pred_all, y_true_all)

    # ✅ (1) ヒストグラム
    plt.figure(figsize=(6, 4))
    plt.hist(angular_errors, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel("Angular Error (°)")
    plt.ylabel("Count")
    plt.title("Angular Error Distribution")
    plt.tight_layout()
    angular_hist_path = OUTPUT_DIR / "angular_error_histogram.png"
    plt.savefig(angular_hist_path)
    plt.show()
    print(f"📁 Saved angular error histogram to: {angular_hist_path}")

    # ✅ (2) 箱ひげ図（外れ値を視覚化）
    plt.figure(figsize=(4, 6))
    plt.boxplot(angular_errors, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightgreen'),
                medianprops=dict(color='red'))
    plt.ylabel("Angular Error (°)")
    plt.title("Angular Error Boxplot")
    plt.tight_layout()
    boxplot_path = OUTPUT_DIR / "angular_error_boxplot.png"
    plt.savefig(boxplot_path)
    plt.show()
    print(f"📁 Saved angular error boxplot to: {boxplot_path}")

    # ✅ (3) 統計指標の出力
    mean_angle = np.mean(angular_errors)
    median_angle = np.median(angular_errors)
    max_angle = np.max(angular_errors)

    print("\n📏 Angular Error Stats (°):")
    print(f"Mean Angular Error: {mean_angle:.4f}°")
    print(f"Median Angular Error: {median_angle:.4f}°")
    print(f"Max Angular Error: {max_angle:.4f}°")

    # ✅ 散布図①：予測ベクトル vs 正解ベクトル（線で誤差可視化）
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    for pred, true in zip(y_pred_all, y_true_all):
        ax.plot([true[0], pred[0]], [true[1], pred[1]], [true[2], pred[2]],
                color='gray', alpha=0.5)  # 誤差を線で
    ax.scatter(y_true_all[:, 0], y_true_all[:, 1], y_true_all[:, 2], color='blue', label='True', alpha=0.6)
    ax.scatter(y_pred_all[:, 0], y_pred_all[:, 1], y_pred_all[:, 2], color='red', label='Predicted', alpha=0.6)

    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    ax.set_title("RGB Vector Scatter (True vs Predicted)")
    ax.legend()
    plt.tight_layout()
    scatter3d_path = OUTPUT_DIR / "rgb_vector_scatter.png"
    plt.savefig(scatter3d_path)
    plt.show()
    print(f"📁 Saved RGB vector scatter plot to: {scatter3d_path}")


    # ✅ 散布図②：サンプルごとの角度誤差
    plt.figure(figsize=(8, 4))
    plt.scatter(np.arange(len(angular_errors)), angular_errors, c='orange', alpha=0.6)
    plt.axhline(np.mean(angular_errors), color='red', linestyle='--', label=f"Mean = {np.mean(angular_errors):.2f}°")
    plt.xlabel("Sample Index")
    plt.ylabel("Angular Error (°)")
    plt.title("Angular Error per Sample")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    scatter_angle_path = OUTPUT_DIR / "angular_error_scatter.png"
    plt.savefig(scatter_angle_path)
    plt.show()
    print(f"📁 Saved angular error scatter plot to: {scatter_angle_path}")

if __name__ == "__main__":
    main()