# eval_auroc_batch.py

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def load_gray_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"❌ 找不到圖檔：{path}")
    return img

def compute_auroc(anomaly_map, mask):
    anomaly_map = cv2.resize(anomaly_map, (mask.shape[1], mask.shape[0]))
    anomaly_map = anomaly_map.astype(np.float32)
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)

    mask = (mask > 127).astype(np.uint8)
    anomaly_flat = anomaly_map.flatten()
    mask_flat = mask.flatten()

    auroc = roc_auc_score(mask_flat, anomaly_flat)
    mean_score = anomaly_flat.mean()
    return auroc, mean_score

def batch_eval(anomaly_dir="eval_output", mask_dir="data"):
    results = []
    for fname in os.listdir(anomaly_dir):
        if not fname.endswith("_anomaly_after.jpg"):
            continue

        name = fname.replace("_anomaly_after.jpg", "")
        anomaly_path = os.path.join(anomaly_dir, fname)

        # 搜尋 mask：從 mask_dir 下所有子資料夾找出符合的 mask
        mask_path = None
        for root, _, files in os.walk(mask_dir):
            for f in files:
                if f == f"{name}_mask.png":
                    mask_path = os.path.join(root, f)
                    break
            if mask_path:
                break

        if not mask_path:
            print(f"⚠️ 找不到 {name} 的 mask，略過")
            continue

        try:
            anomaly = load_gray_image(anomaly_path)
            mask = load_gray_image(mask_path)
            auroc, mean_score = compute_auroc(anomaly, mask)
            results.append({"Image": name, "AUROC": auroc, "MeanScore": mean_score})
            print(f"✅ {name} | AUROC: {auroc:.4f} | Mean: {mean_score:.4f}")
        except Exception as e:
            print(f"❌ {name} 處理失敗：{e}")

    # 儲存成 CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv("auroc_result.csv", index=False)
        print("\n📄 已儲存結果至 auroc_result.csv")
    else:
        print("❗ 沒有成功比對任何 anomaly map 和 mask")

if __name__ == "__main__":
    batch_eval()
