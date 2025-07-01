import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from load_dataset import load_dataset
from MLPModel import MLPModel, mse_chromaticity_loss, train_one_epoch, evaluate
from config import HISTOGRAM_RG_GB_DIR,VAL_HIST_RG_GB_DIR,REAL_RGB_JSON_PATH,EPOCHS, OUTPUT_DIR, BATCH_SIZE, LEARNING_RATE, DEVICE, set_seed

def main():
    set_seed() 
    
    # 1. データ読み込み
    X_train, y_train_df = load_dataset(HISTOGRAM_RG_GB_DIR, REAL_RGB_JSON_PATH)
    X_val, y_val_df = load_dataset(VAL_HIST_RG_GB_DIR, REAL_RGB_JSON_PATH)
    # 出力がX= numpy Y=df

    # 2. Tensorに変換
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train_df[["R", "G", "B"]].values, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val_df[["R", "G", "B"]].values, dtype=torch.float32)


    # 3. TensorDataset作成
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # 4. DataLoader作成
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)

    # 5. モデル定義
    model = MLPModel(input_dim=X_train.shape[1], hidden_dim=256, output_dim=2)
    model.to(DEVICE)
    # SGDオプティマイザで学習
    optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE)
    # 損失関数はクロマティシティ座標のMSE
    loss_fn = mse_chromaticity_loss


    # 学習記録用リスト
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
    
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss = evaluate(model, val_loader, loss_fn)
        print(f"Epoch {epoch+1:02d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # ログ保存
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # 7. モデル保存
    torch.save(model.state_dict(), OUTPUT_DIR / 'mlp_model.pth')

    # 8. 学習曲線の可視化
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()