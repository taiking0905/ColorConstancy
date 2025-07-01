import torch
from torch.utils.data import DataLoader, TensorDataset

from load_dataset import load_dataset
from MLPModel import MLPModel, mse_chromaticity_loss, evaluate
from config import TEST_HIST_DIR,REAL_RGB_JSON_PATH,OUTPUT_DIR, DEVICE


def main():
    # 1. テストデータの読み込み
    X_test_df, y_test_df = load_dataset(TEST_HIST_DIR, REAL_RGB_JSON_PATH)
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
            x = X_test[i].unsqueeze(0)  # shape: (1, input_dim)
            pred = model(x)[0]          # クロマティシティ座標 (r, g)

            # モデル出力（r_pred, g_pred）
            r_pred, g_pred = pred[0].item(), pred[1].item()

            # 正解ラベル（r_true, g_true）はすでに比率
            r_true, g_true = y_test[i].numpy()

            print(f"{i+1}: Pred (r, g): ({r_pred:.4f}, {g_pred:.4f}) | True (r, g): ({r_true:.4f}, {g_true:.4f})")


if __name__ == "__main__":
    main()