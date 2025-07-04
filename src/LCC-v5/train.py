import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from load_dataset import load_dataset
from HistogramDataset import HistogramDataset
from ResNetModel import ResNetModel, angular_loss, train_one_epoch, evaluate
from config import get_base_dir, HISTOGRAM_RG_GB_DIR, VAL_HIST_RG_GB_DIR, REAL_RGB_JSON_PATH, EPOCHS, OUTPUT_DIR, BATCH_SIZE, LEARNING_RATE, WEIGHT,ERASE_PROB, ERASE_SIZE, DEVICE, set_seed, ACCUMULATION_STEPS


def main():
    set_seed() 
    base_dir = get_base_dir()
    print("Base dir:", base_dir)
    print(torch.cuda.is_available())  # TrueならOK
    print(torch.cuda.get_device_name())  # GPU名が出る
    
    # 1. データ読み込み
    X_train, y_train_df = load_dataset(HISTOGRAM_RG_GB_DIR, REAL_RGB_JSON_PATH)
    X_val, y_val_df = load_dataset(HISTOGRAM_RG_GB_DIR, REAL_RGB_JSON_PATH)
    # 出力がX= numpy Y=df
    
    # 2. Tensorに変換
    y_train = y_train_df[["R", "G", "B"]].values
    y_val = y_val_df[["R", "G", "B"]].values
    
    print(f"X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}")
    print(f"X_val.shape = {X_val.shape}, y_val.shape = {y_val.shape}")

    # 3. TensorDataset作成（ここで erase 機能を組み込む）
    train_dataset = HistogramDataset(X_train, y_train, erase_prob=ERASE_PROB, erase_size = ERASE_SIZE)
    val_dataset = HistogramDataset(X_val, y_val)

    # 4. DataLoader作成 pin_memory=Trueこれを使うとGPUへの転送が速くなる
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)


    # 5. モデル定義
    model = ResNetModel().to(DEVICE)
    try:
        model = torch.compile(model, backend="eager")
    except Exception as e:
        print(f"torch.compile failed: {e}")
    # Adamオプティマイザで学習
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT)


    # 損失関数はRGBベクトル間の角度誤差
    loss_fn = angular_loss


    # 学習記録用リスト
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, accumulation_steps=ACCUMULATION_STEPS)

        val_loss = evaluate(model, val_loader, loss_fn)
        print(f"Epoch {epoch+1:02d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # ログ保存
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # 7. モデル保存
    torch.save(model.state_dict(), OUTPUT_DIR / 'resnet_model.pth')
    plt.savefig(OUTPUT_DIR / 'loss_curve.png')

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