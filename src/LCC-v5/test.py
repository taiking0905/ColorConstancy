import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from load_dataset import load_dataset
from HistogramDataset import HistogramDataset
from ResNetModel import ResNetModel, angular_loss, evaluate
from config import HISTOGRAM_RG_GB_DIR, TEST_HIST_DIR, REAL_RGB_JSON_PATH, OUTPUT_DIR,OUTPUT_DIM, BATCH_SIZE, DEVICE, set_seed


def main():
    set_seed()
    # 1. テストデータの読み込み
    X_test, y_test_df = load_dataset(HISTOGRAM_RG_GB_DIR, REAL_RGB_JSON_PATH)
    
    y_test = y_test_df[["R", "G" , "B"]].values

    print(f"X_test.shape = {X_test.shape}, y_test.shape = {y_test.shape}")

    val_dataset = HistogramDataset(X_test, y_test)

    # 2. モデル構造の再定義（構造は学習と同じに！）
    model = ResNetModel(output_dim=OUTPUT_DIM)
    model.load_state_dict(torch.load(OUTPUT_DIR / 'resnet_model.pth'))  # 学習済みモデルの読み込み
    model.to(DEVICE)  
    model.eval()

    # DataLoader作成 pin_memory=Trueこれを使うとGPUへの転送が速くなる
    test_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=2)


    # 3. 評価実行（RGBベクトル間角度誤差）
    test_loss = evaluate(model, test_loader, angular_loss)
    print(f"📊 Test Loss = {test_loss:.4f}")

    # 4. 予測と実際のRGB値の表示（5件だけ例として）
    print("\n🎨 Prediction vs Actual (first 5 samples):")
    
    num_samples = min(5, len(X_test))  # データ数に合わせる
    with torch.no_grad():
        for i in range(num_samples):
            x = torch.tensor(X_test[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pred = model(x)[0].cpu()          #  (r, g, b)
            pred = pred / torch.norm(pred)

            # モデル出力（r_pred, g_pred, b_pred）
            r_pred, g_pred, b_pred= pred[0].item(), pred[1].item(), pred[2].item()

            # 正解ラベル（r_true, g_true, b_true）はすでに比率
            r_true, g_true, b_true = y_test[i]

            print(f"{i+1}: Pred (r, g, b): ({r_pred:.4f}, {g_pred:.4f}, {b_pred:.4f}) | True (r, g, b): ({r_true:.4f}, {g_true:.4f}, {b_true:.4f})")


if __name__ == "__main__":
    main()