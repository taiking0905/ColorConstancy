# Color Constancy

本プロジェクトは、機械学習をして色恒常性を手に入れる研究です。

---

## ✅ 環境構築手順

### Python仮想環境の作成

```bash
python -m venv ColorConstancy_env   

.\ColorConstancy_env\Scripts\Activate   

pip install -r requirements.txt

```"E:\ColorConstancy\histogram"

## 🔧 インストール手順（GPU対応）

このプロジェクトでは PyTorch の CUDA 対応版（12.8）を使用しています。  
以下のコマンドでインストールしてください：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128


```bash

project_root/
├── datapre               # 試し実行用データ
├── src/                  # Vscode作業ファイル
├──├──LCC-v0/
├──├── config.py          # 訓練やモデルのハイパーパラメータ設定
├──├── load_dataset.py    # データセットの読み込みロジック
├──├── MLPModel.py        # MLPモデルの定義
├──├── train.py           # 学習スクリプト
├──├── test.py            # テストスクリプト
├──├── outputs/           # 学習モデル・ログの保存先
├──└── __pycache__/       # Pythonのキャッシュ（Git管理不要）
|
├──├──pretreatment.py     #前処理
├── ColorConstancy_env/   # 仮想環境（この中はGit管理しない）
├── requirements.txt      # 使用ライブラリ一覧
├── real_rgb.json         # 教師データ
└── README.md             # このファイル
````

```bash

USB/ColorConstancy
├── /datasets 
├── /histogram       
├── /enddatasets
└── 
````
--- 
進捗報告:   
https://field-motorcycle-315.notion.site/2492c8c0fa7c8083a14ff4999e612917

---
