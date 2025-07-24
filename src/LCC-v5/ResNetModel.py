import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from config import DROPOUT, OUTPUT_DIM, DEVICE

# ResNet18をベースにしたカラー推定モデル（入力: 1ch, 出力: RGBベクトル）
class ResNetModel(nn.Module):
    def __init__(self, output_dim = OUTPUT_DIM, dropout_rate = DROPOUT):
        super().__init__()
        model = resnet18(weights=None)

        # 入力チャンネルを1ch（グレースケールやヒストグラム画像）に変更
        model.conv1 = nn.Conv2d(
            in_channels=1,       # 入力は1チャネル
            out_channels=64,     # 出力は64チャネル（ResNet標準）
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
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

def train_one_epoch(model, loader, optimizer, loss_fn, accumulation_steps=1):
    # iter_start = time.time()
    # print(f"⏱️ First iter(loader): {time.time() - iter_start:.3f} sec")

    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, (X_batch, y_batch) in enumerate(loader):
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)
        loss = loss / accumulation_steps  # 🔑 勾配スケーリング

        loss.backward()
        end_event.record()

        # ✅ accumulation_steps回ごとにパラメータ更新
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    average_loss = total_loss / len(loader)
    return average_loss




# 🔍 評価関数（検証・テスト用）
def evaluate(model, loader, loss_fn):
    model.eval()  
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
