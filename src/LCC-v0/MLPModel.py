import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, hidden1_dim)
        self.hidden2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.output = nn.Linear(hidden2_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        h1 = self.sigmoid(self.hidden1(x))
        h2 = self.sigmoid(self.hidden2(h1))
        out = self.sigmoid(self.output(h2)) 
        return out

def euclidean_loss(pred, target):
    return torch.sqrt(((pred - target) ** 2).sum(dim=1)).mean()

def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()  # モデルを訓練モードに設定
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(next(model.parameters()).device)  # 追加
        y_batch = y_batch.to(next(model.parameters()).device)  # 追加
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
            X_batch = X_batch.to(next(model.parameters()).device)  # 追加
            y_batch = y_batch.to(next(model.parameters()).device)  # 追加
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch[:, :2])
            total_loss += loss.item()

    average_loss = total_loss / len(loader)
    return average_loss