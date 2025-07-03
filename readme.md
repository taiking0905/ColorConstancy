# Color Constancy

本プロジェクトは、機械学習をして色恒常性を手に入れる研究です。

---

## ✅ 環境構築手順

### 1. Python仮想環境の作成

```bash
python -m venv ColorConstancy_env   

.\ColorConstancy_env\Scripts\Activate   

pip install -r requirements.txt

dir /a-d /b "E:\ColorConstancy\histogram" | find /c /v ""
```"E:\ColorConstancy\histogram"

### 2.仮想環境起動方法

```bash
.\ColorConstancy_env\Scripts\Activate   

```

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