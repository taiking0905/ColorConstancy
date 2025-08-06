import json
import os
import pandas as pd
import numpy as np

def load_dataset(csv_dir, json_path):
    # 1. JSON読み込みと辞書化
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    rgb_dict = {item["filename"]: item["real_rgb"] for item in json_data}

    # 2. 特徴量とラベルの蓄積
    X_list = []
    y_list = []

    for filename in os.listdir(csv_dir):
        if not filename.endswith("_masked.csv"):
            continue

        base_id = filename.replace("_masked.csv", "")  # 例: 8D5U5524

        if base_id not in rgb_dict:
            print(f"Warning: {base_id} not in JSON, skipping")
            continue

        csv_path = os.path.join(csv_dir, filename)
        df = pd.read_csv(csv_path, header=None)

        # 特徴量は1行と仮定してflatten
        features = df.values.flatten()

        X_list.append(features)

        # real_rgbからr, gの比率を計算
        R, G, B = rgb_dict[base_id]
        total = R + G + B if R + G + B != 0 else 1e-6  # 0割防止

        r_ratio = R / total
        g_ratio = G / total
        b_ratio = B / total

        y_list.append([r_ratio, g_ratio, b_ratio])

    # DataFrameに変換
    X = pd.DataFrame(X_list)
    y = pd.DataFrame(y_list, columns=["r_ratio", "g_ratio", "b_ratio"])

    # NumPy配列に変換（学習用など）
    X_np = X.to_numpy().astype(np.float32)  # shape: (N, 1250)
    y_np = y.to_numpy().astype(np.float32)  # shape: (N, 3)

    return X_np, y_np
