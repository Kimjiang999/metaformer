# eval_auroc.py

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

def load_gray_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"找不到圖檔：{path}")
    return img

def compute_auroc(anomaly_map, mask):
    # resize anomaly_map to match mask size
    anomaly_map = cv2.resize(anomaly_map, (mask.shape[1], mask.shape[0]))

    # 正規化 anomaly map 為 0~1
    anomaly_map = anomaly_map.astype(np.float32)
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)

    # 將 mask 轉為二值 (白色為 1、黑為 0)
    mask = (mask > 127).astype(np.uint8)

    # 展平為一維向量
    anomaly_flat = anomaly_map.flatten()
    mask_flat = mask.flatten()

    auroc = roc_auc_score(mask_flat, anomaly_flat)
    mean_score = anomaly_flat.mean()

    return auroc, mean_score

if __name__ == "__main__":
    # ✅ 替換為你自己的檔案路徑
    anomaly_map_path = "eval_output/bottle_anomaly_before.jpg"
    mask_path = "data/bottle_eval/bottle_mask.png"

    print(f"🔍 正在評估：{anomaly_map_path} 對照 {mask_path}")

    anomaly = load_gray_image(anomaly_map_path)
    mask = load_gray_image(mask_path)

    auroc, mean_score = compute_auroc(anomaly, mask)

    print(f"✅ AUROC: {auroc:.4f}")
    print(f"✅ 平均 anomaly score: {mean_score:.4f}")
