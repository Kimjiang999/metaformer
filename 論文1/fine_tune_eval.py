# fine_tune_eval.py

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from metaformer.metaformer_model import Metaformer
from utils.data_loader import load_all_classes, sample_meta_task

def preprocess_batch(images):
    return tf.convert_to_tensor(images, dtype=tf.float32)

def repeat_saliency(images):
    return tf.repeat(images[..., :1], repeats=3, axis=-1)

def visualize_comparison(query_img, saliency, recon_before, recon_after, prefix="output"):
    anomaly_before = np.abs(query_img - recon_before).mean(axis=-1)
    anomaly_after = np.abs(query_img - recon_after).mean(axis=-1)

    heat_before = cv2.applyColorMap(cv2.normalize(anomaly_before, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
    heat_after = cv2.applyColorMap(cv2.normalize(anomaly_after, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)

    input_bgr = cv2.cvtColor((query_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    overlay_before = cv2.addWeighted(input_bgr, 0.6, heat_before, 0.4, 0)
    overlay_after = cv2.addWeighted(input_bgr, 0.6, heat_after, 0.4, 0)

    os.makedirs("eval_output", exist_ok=True)
    cv2.imwrite(f"eval_output/{prefix}_input.jpg", input_bgr)
    cv2.imwrite(f"eval_output/{prefix}_saliency.jpg", (saliency * 255).astype(np.uint8))
    cv2.imwrite(f"eval_output/{prefix}_recon_before.jpg", (recon_before * 255).astype(np.uint8))
    cv2.imwrite(f"eval_output/{prefix}_recon_after.jpg", (recon_after * 255).astype(np.uint8))
    cv2.imwrite(f"eval_output/{prefix}_anomaly_before.jpg", heat_before)
    cv2.imwrite(f"eval_output/{prefix}_anomaly_after.jpg", heat_after)
    cv2.imwrite(f"eval_output/{prefix}_overlay_before.jpg", overlay_before)
    cv2.imwrite(f"eval_output/{prefix}_overlay_after.jpg", overlay_after)

    print(f"✅ 已儲存比較圖至 eval_output/{prefix}_*.jpg")

# === 主程式 ===
if __name__ == "__main__":
    print("🚀 Fine-tuning 評估開始...")

    # 載入資料
    class_data = load_all_classes("data", image_size=(64, 64), max_per_class=20)
    class_name, support, query = sample_meta_task(class_data, n_support=5, n_query=1)

    print(f"📌 評估類別：{class_name}")

    support = preprocess_batch(support)
    query = preprocess_batch(query)

    # 初始化主模型
    base_model = Metaformer(embed_dim=32)
    base_model.build(input_shape=[(1, 64, 64, 3), (1, 64, 64, 1)])  # 👈 這一行是關鍵！

    # 載入權重
    if os.path.exists("metaformer_meta_trained.weights.h5"):
        base_model.load_weights("metaformer_meta_trained.weights.h5")
        print("📥 已載入 meta-trained 模型參數")
    else:
        print("⚠️ 找不到儲存模型，將使用未訓練模型")

    # 推論 query 前重建
    saliency_q = repeat_saliency(query)
    recon_before = base_model.forward(query, saliency_q, training=False)[0].numpy()

    # 微調
    for _ in range(3):
        with tf.GradientTape() as tape:
            saliency_s = repeat_saliency(support)
            recon = base_model.forward(support, saliency_s, training=True)
            loss = tf.keras.losses.MeanSquaredError()(support, recon)
        grads = tape.gradient(loss, base_model.trainable_variables)
        tf.keras.optimizers.SGD(learning_rate=0.01).apply_gradients(zip(grads, base_model.trainable_variables))

    # 推論微調後結果
    recon_after = base_model.forward(query, saliency_q, training=False)[0].numpy()
    query_img = query[0].numpy()
    saliency_img = saliency_q[0].numpy()

    # 視覺化
    visualize_comparison(query_img, saliency_img, recon_before, recon_after, prefix=class_name)
