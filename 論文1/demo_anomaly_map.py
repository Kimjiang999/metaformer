# demo_anomaly_map.py

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from metaformer.metaformer_model import Metaformer
from metaformer.instance_prior import get_instance_prior

def load_image(path, target_size=(64, 64)):
    img = cv2.imread(path)
    img = cv2.resize(img, target_size)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = tf.convert_to_tensor(img_rgb / 255.0, dtype=tf.float32)
    return tf.expand_dims(img_tensor, axis=0), img_rgb  # [1, H, W, 3], 原始 RGB

def visualize_and_save(input_img, saliency, recon_img, anomaly_map, output_prefix="result"):
    os.makedirs("output", exist_ok=True)

    # Save saliency and anomaly heatmap
    cv2.imwrite(f"output/{output_prefix}_input.jpg", cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"output/{output_prefix}_saliency.jpg", saliency)
    cv2.imwrite(f"output/{output_prefix}_reconstruction.jpg", cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR))

    # Normalize and save anomaly heatmap
    norm_map = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
    cv2.imwrite(f"output/{output_prefix}_anomaly_map.jpg", heatmap)

    # 疊圖（透明重疊）
    overlay = cv2.addWeighted(cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)
    cv2.imwrite(f"output/{output_prefix}_overlay.jpg", overlay)

    # 顯示在畫面上
    fig, axs = plt.subplots(1, 5, figsize=(18, 4))
    axs[0].imshow(input_img)
    axs[0].set_title("Input Image")

    axs[1].imshow(saliency, cmap='gray')
    axs[1].set_title("Saliency Map")

    axs[2].imshow(recon_img)
    axs[2].set_title("Reconstruction")

    axs[3].imshow(anomaly_map, cmap='hot')
    axs[3].set_title("Anomaly Map")

    axs[4].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axs[4].set_title("Overlay")

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# === 主程式 ===
if __name__ == "__main__":
    img_path = "data/bottle_eval/000.png"  # ← 改成你放的圖片檔名

    # 載入與處理圖片
    input_tensor, input_rgb = load_image(img_path)
    saliency_map = get_instance_prior(img_path)
    saliency_map = cv2.resize(saliency_map, (64, 64))
    saliency_tensor = tf.convert_to_tensor(saliency_map / 255.0, dtype=tf.float32)
    saliency_tensor = tf.expand_dims(saliency_tensor, axis=0)
    saliency_tensor = tf.expand_dims(saliency_tensor, axis=-1)

    # 模型推論
    model = Metaformer()
    recon = model.forward(input_tensor, saliency_tensor, training=False)
    recon_np = recon[0].numpy()

    # 產生 anomaly map
    anomaly_map = np.abs(input_tensor[0].numpy() - recon_np).mean(axis=-1)

    # 儲存與顯示
    visualize_and_save(
        input_img=(input_rgb),
        saliency=saliency_map,
        recon_img=(recon_np * 255).astype(np.uint8),
        anomaly_map=anomaly_map,
        output_prefix="demo"
    )
