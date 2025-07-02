import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from config import DEVICE 

class ResNetModel(nn.Module):
    def __init__(self, output_dim =3):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, output_dim)  # 出力調整（ここでは色度座標2次元）

    def forward(self, x):
        return self.model(x)

#もう一つ角度そのものを損失にする可能性あり
def angular_loss(pred, target):
    """
    pred: (N, 3) RGBベクトル（モデルの出力）
    target: (N, 3) RGBベクトル（正解ラベル）
    """
    pred_norm = F.normalize(pred, dim=1)     # L2正規化
    target_norm = F.normalize(target, dim=1) # L2正規化

    cos_sim = (pred_norm * target_norm).sum(dim=1)  # 各バッチでcosθ
    loss = 1 - cos_sim  # 角度差に応じた損失
    return loss.mean()

def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()  # モデルを訓練モードに設定
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()               # 勾配をリセット
        pred = model(X_batch)               # 順伝播
        loss = loss_fn(pred, y_batch)       # 損失計算
        loss.backward()                     # 逆伝播
        optimizer.step()                    # パラメータ更新

        total_loss += loss.item()           # 損失を蓄積

    average_loss = total_loss / len(loader)  # バッチ数で割る
    return average_loss

def evaluate(model, loader, loss_fn):
    model.eval()  # 評価モードに切り替え
    total_loss = 0.0

    with torch.no_grad():  # 勾配を計算しない（メモリ節約＆高速化）
        
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            total_loss += loss.item()

    average_loss = total_loss / len(loader)
    return average_loss