from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

# ======== Matplotlib設定 ========
matplotlib.use('TkAgg')  # TkAggバックエンドを使用（ウィンドウ移動対応）

# ======== グローバル定数 ========

# データパスのベース
BASE_PATH = Path("D:/ColorConstancy")

# カメラ設定（black/whiteレベル）
BLACK_LEVEL = 129
WHITE_LEVEL = 4095

# ======== ディレクトリ構成 ========

def setup_directories():
    """
    必要なディレクトリを作成し、パス辞書を返す。
    real_rgb.json の存在チェックも行う。
    """
    dirs = {
        "INPUT": BASE_PATH / "datasets",
        "MASK": BASE_PATH / "masking",
        "HIST": BASE_PATH / "histogram",
        # "TEACHER": BASE_PATH / "teacher",
        # "TEACHER_HIST": BASE_PATH / "teacherhist",
        "REAL_RGB_JSON": BASE_PATH / "real_rgb.json",
        "END": BASE_PATH / "enddatasets"
    }

    # ディレクトリ作成（JSONファイルは除外）
    for key, path in dirs.items():
        if key != "REAL_RGB_JSON":
            path.mkdir(parents=True, exist_ok=True)

    # JSONファイルの存在確認
    if not dirs["REAL_RGB_JSON"].exists():
        print("⚠ real_rgb.json が存在しません。先に作成してください。")

    return dirs

# ======== ウィンドウ配置関数 ========

def move_figure(fig, x: int, y: int):
    """
    Matplotlibのウィンドウ位置を画面上の指定座標(x, y)に移動する。
    """
    backend = plt.get_backend()
    if backend == 'TkAgg':
        fig.canvas.manager.window.wm_geometry(f"+{x}+{y}")
    else:
        print(f"ウィンドウ移動はこのバックエンドでは未対応: {backend}")
