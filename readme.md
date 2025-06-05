# Dataset Preprocessing & Histogram Project

本プロジェクトは、Canon 568データセットに対するマスク処理とヒストグラム抽出、および機械学習を行うためのPython環境です。

---

## ✅ 環境構築手順

### 1. Python仮想環境の作成

```bash
python -m venv ColorConstancy_env   

.\ColorConstancy_env\Scripts\Activate   

pip install -r requirements.txt

```

### 2.仮想環境起動方法

```bash
.\ColorConstancy_env\Scripts\Activate   

jupyter lab 

```

```bash

project_root/
├── src/            # Jupyter作業ファイル
├── cc_env/               # 仮想環境（この中はGit管理しない）
├── requirements.txt      # 使用ライブラリ一覧
└── README.md             # このファイル
````