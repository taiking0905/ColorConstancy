import numpy as np
import torch

class HistogramDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, erase_prob=0.0, erase_size=0.0):
        self.X = X
        self.y = y
        self.erase_prob = erase_prob
        self.erase_size = erase_size

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 画像取得 
        x = self.X[idx]
        y = self.y[idx]

        # eraseを確率で適用
        if np.random.rand() < self.erase_prob:
            x = self.random_erase(x)

        # Tensorに変換（[1, 224, 224] で CNN対応）
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

    def random_erase(self, img):
        x = np.random.randint(0, 224 - self.erase_size)
        y = np.random.randint(0, 224 - self.erase_size)
        img[y:y+self.erase_size, x:x+self.erase_size] = 0
        return img

