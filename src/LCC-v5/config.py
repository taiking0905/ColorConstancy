# config.py
from pathlib import Path
import torch
import numpy as np
import random

# -------------------------------
# パス設定
# -------------------------------


BASE_DIR = Path("D:/ColorConstancy")
LCC_DIR = Path(__file__).resolve().parent
HISTOGRAM_RG_GB_DIR = BASE_DIR / "histogram_rg_gb"
VAL_HIST_RG_GB_DIR = BASE_DIR / "valhist_rg_gb"
REAL_RGB_JSON_PATH = BASE_DIR / "real_rgb.json"
TEST_HIST_DIR = BASE_DIR / "test"
OUTPUT_DIR = LCC_DIR / "outputs"
# -------------------------------
# ハイパーパラメータ
# -------------------------------
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# -------------------------------
# デバイス設定
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# ランダムシード固定
# -------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
