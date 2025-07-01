import torch
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()  # 出力を0〜1に制限

    def forward(self, x):
        # 活性化関数は使用せず、線形変換のみ
        h = self.hidden(x)  # 隠れ層（線形のみ）
        out = self.output(h)  # 出力層（線形）
        out = self.sigmoid(out)  # 0〜1に制限
        return out


def euclidean_loss(pred, target):
    return torch.sqrt(((pred - target) ** 2).sum(dim=1)).mean()

def mse_chromaticity_loss(pred, target, eps=1e-8):
    # クロマティシティ座標に変換：r = R/(R+G+B), g = G/(R+G+B)
    pred_sum = pred.sum(dim=1, keepdim=True) + eps
    target_sum = target.sum(dim=1, keepdim=True) + eps

    pred_chroma = pred[:, :2] / pred_sum  # (r, g)
    target_chroma = target[:, :2] / target_sum

    # クロマティシティ座標のMSE損失を計算
    loss = ((pred_chroma - target_chroma) ** 2).mean()
    return loss

def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()  # モデルを訓練モードに設定
    total_loss = 0.0

    for X_batch, y_batch in loader:
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
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            total_loss += loss.item()

    average_loss = total_loss / len(loader)
    return average_loss