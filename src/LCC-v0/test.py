import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

from load_dataset import load_dataset
from MLPModel import MLPModel, mse_chromaticity_loss, evaluate
from config import TEST_DIR,REAL_RGB_JSON_PATH,OUTPUT_DIR, DEVICE


def main():
    # 1. テストデータの読み込み
    X_test_df, y_test_df = load_dataset(TEST_DIR, REAL_RGB_JSON_PATH)
    X_test = torch.tensor(X_test_df.values, dtype=torch.float32)
    y_test = torch.tensor(y_test_df[["r_ratio", "g_ratio"]].values, dtype=torch.float32)

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

            # 正解ラベル（r_true, g_true）はすでに比率
            r_true, g_true = y_test[i].numpy()

            print(f"{i+1}: Pred (r, g): ({r_pred:.4f}, {g_pred:.4f}) | True (r, g): ({r_true:.4f}, {g_true:.4f})")
        # 5. 可視化: 散布図と誤差ヒストグラム
        y_pred_list = []
        y_true_list = []

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(DEVICE)
                preds = model(x_batch).cpu().numpy()
                y_pred_list.append(preds)
                y_true_list.append(y_batch.numpy())

        y_pred_all = np.vstack(y_pred_list)  # shape: (N, 2)
        y_true_all = np.vstack(y_true_list)  # shape: (N, 2)

        # (1) 散布図
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true_all[:, 0], y_true_all[:, 1], label='True', alpha=0.6, c='blue', s=30)
        plt.scatter(y_pred_all[:, 0], y_pred_all[:, 1], label='Predicted', alpha=0.6, c='red', s=30, marker='x')
        plt.xlabel("r_ratio")
        plt.ylabel("g_ratio")
        plt.title("Chromaticity: Predicted vs True")
        plt.legend()
        plt.grid(True)

        # 軸の範囲を0〜1に固定（これが追加点）
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        plt.tight_layout()
        scatter_path = OUTPUT_DIR / "chromaticity_scatter.png"
        plt.savefig(scatter_path)
        print(f"📁 Saved scatter plot to: {scatter_path}")
        plt.show()


        # (2) 誤差ヒストグラム（ユークリッド距離）
        errors = np.linalg.norm(y_pred_all - y_true_all, axis=1)
        plt.figure(figsize=(6, 4))
        plt.hist(errors, bins=30, color='gray', edgecolor='black')
        plt.xlabel("Euclidean Error (r,g)")
        plt.ylabel("Count")
        plt.title("Prediction Error Distribution")
        plt.tight_layout()
        hist_path = OUTPUT_DIR / "error_histogram.png"
        plt.savefig(hist_path)
        print(f"📁 Saved histogram to: {hist_path}")
        plt.show()
        # (3) 統計情報も出力
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        max_error = np.max(errors)
        print(f"\n📏 Error Stats:")
        print(f"Mean Error: {mean_error:.4f}")
        print(f"Median Error: {median_error:.4f}")
        print(f"Max Error: {max_error:.4f}")


if __name__ == "__main__":
    main()