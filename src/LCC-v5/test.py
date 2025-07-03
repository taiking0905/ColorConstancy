import torch
from torch.utils.data import DataLoader, TensorDataset

from load_dataset import load_dataset
from ResNetModel import ResNetModel, angular_loss, evaluate
from config import HISTOGRAM_RG_GB_DIR,TEST_HIST_DIR,REAL_RGB_JSON_PATH,OUTPUT_DIR, DEVICE


def main():
    # 1. テストデータの読み込み
    X_test_df, y_test_df = load_dataset(HISTOGRAM_RG_GB_DIR, REAL_RGB_JSON_PATH)
    X_test = torch.tensor(X_test_df, dtype=torch.float32)
    y_test = torch.tensor(y_test_df[["R", "G" , "B"]].values, dtype=torch.float32)

    # 2. モデル構造の再定義（構造は学習と同じに！）
    model = ResNetModel(output_dim=3)
    model.load_state_dict(torch.load(OUTPUT_DIR / 'resnet_model.pth'))  # 学習済みモデルの読み込み
    model.to(DEVICE)  
    model.eval()

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 3. 評価実行（RGBベクトル間角度誤差）
    test_loss = evaluate(model, test_loader, angular_loss)
    print(f"📊 Test Loss = {test_loss:.4f}")

    # 4. 予測と実際のRGB値の表示（5件だけ例として）
    print("\n🎨 Prediction vs Actual (first 5 samples):")
    
    num_samples = min(5, len(X_test))  # データ数に合わせる
    with torch.no_grad():
        for i in range(num_samples):
            x = X_test[i].unsqueeze(0).to(DEVICE)  # shape: (1, input_dim)
            pred = model(x)[0].cpu()          #  (r, g, b)
            pred = pred / torch.norm(pred)

            # モデル出力（r_pred, g_pred, b_pred）
            r_pred, g_pred, b_pred= pred[0].item(), pred[1].item(), pred[2].item()

            # 正解ラベル（r_true, g_true, b_true）はすでに比率
            r_true, g_true, b_true = y_test[i].numpy()

            print(f"{i+1}: Pred (r, g, b): ({r_pred:.4f}, {g_pred:.4f}, {b_pred:.4f}) | True (r, g, b): ({r_true:.4f}, {g_true:.4f}, {b_true:.4f})")


if __name__ == "__main__":
    main()