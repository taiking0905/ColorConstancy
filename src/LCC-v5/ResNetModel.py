import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights

from config import DROPOUT, OUTPUT_DIM, DEVICE

# ResNet18をベースにしたカラー推定モデル（入力: 1ch, 出力: RGBベクトル）
class ResNetModel(nn.Module):
    def __init__(self, output_dim = OUTPUT_DIM, dropout_rate = DROPOUT):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)

         # conv1 の重みを1ch用に変換（3ch → 平均で1chへ）
        pretrained_weight = model.conv1.weight
        new_weight = pretrained_weight.mean(dim=1, keepdim=True)

        # 入力チャンネルを1ch（グレースケールやヒストグラム画像）に変更
        model.conv1 = nn.Conv2d(
            in_channels=1,       # 入力は1チャネル
            out_channels=64,     # 出力は64チャネル（ResNet標準）
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        model.conv1.weight.data = new_weight
        
        self.model = model


        # 既存のfc層の前にDropoutを挟む構造に変更
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),          # Dropoutを追加
            nn.Linear(in_features, output_dim)   # 最終出力
        )

        
    def forward(self, x):
        return self.model(x)  # 順伝播（出力は shape: [batch_size, 3]）


# 🔺角度ベースの損失関数（色ベクトルの方向を比較）
def angular_loss(pred, target):
    """
    pred: モデルの出力 (N, 3)
    target: 正解のRGB比率ベクトル (N, 3)
    → 出力ベクトルと正解ベクトルの角度（cos類似度）で誤差を計算
    """
    pred_norm = F.normalize(pred, dim=1)     # 出力をL2正規化（長さを1に）
    target_norm = F.normalize(target, dim=1) # 正解もL2正規化

    cos_sim = (pred_norm * target_norm).sum(dim=1)  # 各ベクトル間のcos類似度
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    
    loss = 1 - cos_sim  # cosθが高い（方向が一致）ほど損失が小さい
    return loss.mean()  # バッチ平均の損失を返す


# 🔁 1エポック分の訓練処理
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()  # 訓練モード（DropoutやBatchNormを有効化）
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()               # 勾配の初期化
        pred = model(X_batch)               # モデル予測（順伝播）
        loss = loss_fn(pred, y_batch)       # 損失を計算
        loss.backward()                     # 勾配計算（逆伝播）
        optimizer.step()                    # パラメータ更新

        total_loss += loss.item()           # バッチごとの損失を蓄積

    average_loss = total_loss / len(loader)  # バッチ数で割って平均損失を算出
    return average_loss


# 🔍 評価関数（検証・テスト用）
def evaluate(model, loader, loss_fn):
    model.eval()  # 評価モード（DropoutやBatchNormを無効化）
    total_loss = 0.0

    with torch.no_grad():  # 勾配を計算しない（推論のみで高速・省メモリ）

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            pred = model(X_batch)               # モデル出力
            loss = loss_fn(pred, y_batch)       # 損失を計算
            total_loss += loss.item()

    average_loss = total_loss / len(loader)     # 全体の平均損失
    return average_loss
