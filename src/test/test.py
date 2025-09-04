import rawpy, cv2, numpy as np, os, glob, json

# === 設定 ===
dirs = r""
pattern = "*.dng"  # DNGを直接読む
# ROIの指定: 画像中央10%をデフォルトに（グレーカードを中央付近に置く想定）
ROI_FRACTION = 0.1
SAVE_PNG = True
JSON_PATH = os.path.join(dirs, "gt_illuminants.json")

results = []

dng_paths = sorted(glob.glob(os.path.join(dirs, pattern)))
for p in dng_paths:
    try:
        name = os.path.splitext(os.path.basename(p))[0]
        with rawpy.imread(p) as raw:
            # 線形RGB（WBなし/ガンマなし/自動明るさなし）
            rgb16 = raw.postprocess(
                use_camera_wb=False,
                no_auto_bright=True,
                output_bps=16,
                gamma=(1, 1),
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD
            )  # HxWx3, uint16, 0..65535（機種により実効レンジは異なる）

        # ROI（中央）
        H, W = rgb16.shape[:2]
        fh = int(H * ROI_FRACTION)
        fw = int(W * ROI_FRACTION)
        y0 = (H - fh) // 2
        x0 = (W - fw) // 2

        roi = rgb16[y0:y0+fh, x0:x0+fw, :].astype(np.float32)

        # クリップ/暗部の除外マスク（上位2%・下位1%を除外の例）
        maxv = roi.max()
        high_th = maxv * 0.98
        low_th  = maxv * 0.01
        mask = (roi[:,:,0] < high_th) & (roi[:,:,1] < high_th) & (roi[:,:,2] < high_th) & \
               (roi[:,:,0] > low_th)  & (roi[:,:,1] > low_th)  & (roi[:,:,2] > low_th)

        # マスクがスカスカならROIをそのまま使う（保険）
        if mask.sum() < roi.shape[0]*roi.shape[1]*0.2:
            mask = np.ones(roi.shape[:2], dtype=bool)

        Rm = roi[:,:,0][mask].mean()
        Gm = roi[:,:,1][mask].mean()
        Bm = roi[:,:,2][mask].mean()

        # 照明ベクトル（グレーカードは等反射と仮定）
        e = np.array([Rm, Gm, Bm], dtype=np.float32)

        # Gで正規化（学習で扱いやすい）
        if Gm <= 1e-6:
            e_norm = e / (np.linalg.norm(e) + 1e-6)  # 代替: L2正規化
            norm_type = "L2"
        else:
            e_norm = e / Gm
            norm_type = "G-norm"

        # WB係数（モデルによってはこちらを正解にすることも）
        # 「e_norm * wb = [1,1,1]」になるように wb = 1 / e_norm
        wb = 1.0 / np.maximum(e_norm, 1e-6)

        # 出力PNG（任意）
        if SAVE_PNG:
            out_png = os.path.join(dirs, f"{name}_linear.png")
            cv2.imwrite(out_png, cv2.cvtColor(rgb16, cv2.COLOR_RGB2BGR))

            # 12bit画像も保存
            white_level = 4095
            black_level = 528
            rgb_12bit = ((rgb16.astype(np.float32) - black_level) / (white_level - black_level) * 4095).clip(0, 4095).astype(np.uint16)
            out_png_12bit = os.path.join(dirs, f"{name}_linear_12bit.png")
            cv2.imwrite(out_png_12bit, cv2.cvtColor(rgb_12bit, cv2.COLOR_RGB2BGR))


        # ログ保持
        rec = {
            "filename": os.path.basename(p),
            "norm_type": norm_type,
            "e_norm_r": float(e_norm[0]),
            "e_norm_g": float(e_norm[1]),
            "e_norm_b": float(e_norm[2]),
            "wb_r": float(wb[0]),
            "wb_g": float(wb[1]),
            "wb_b": float(wb[2]),
            "roi_xywh": [int(x0), int(y0), int(fw), int(fh)]
        }
        results.append(rec)
        print(f"[OK] {name}: e_norm={e_norm}, wb={wb}")

    except Exception as ex:
        print(f"[ERR] {p}: {ex}")

# 保存（JSON）

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Saved: {CSV_PATH}")
print(f"Saved: {JSON_PATH}")
