import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from load_dataset import load_dataset
from ResNetModel import ResNetModel, angular_loss, train_one_epoch, evaluate
from config import HISTOGRAM_RG_GB_DIR,VAL_HIST_RG_GB_DIR,REAL_RGB_JSON_PATH,EPOCHS, OUTPUT_DIR, BATCH_SIZE, LEARNING_RATE, DEVICE, set_seed

def main():
    set_seed() 
    
    # 1. データ読み込み
    X_train, y_train_df = load_dataset(HISTOGRAM_RG_GB_DIR, REAL_RGB_JSON_PATH)
    # X_val, y_val_df = load_dataset(VAL_HIST_RG_GB_DIR, REAL_RGB_JSON_PATH)
    # 出力がX= numpy Y=df

    print(X_train.shape)  # → (N, 1, 448, 224)
    print(y_train_df.head())  # → R, G, B列





if __name__ == "__main__":
    main()