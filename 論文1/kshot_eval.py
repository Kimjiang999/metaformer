# kshot_eval.py

import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from metaformer.metaformer_model import Metaformer
from utils.data_loader import load_all_classes

def preprocess(images):
    return tf.convert_to_tensor(images, dtype=tf.float32)

def repeat_saliency(images):
    return tf.repeat(images[..., :1], repeats=3, axis=-1)

def fine_tune_and_infer(model, support, query, steps=3):
    support_sal = repeat_saliency(support)
    for _ in range(steps):
        with tf.GradientTape() as tape:
            recon = model.forward(support, support_sal, training=True)
            loss = tf.keras.losses.MeanSquaredError()(support, recon)
        grads = tape.gradient(loss, model.trainable_variables)
        tf.keras.optimizers.SGD(learning_rate=0.01).apply_gradients(zip(grads, model.trainable_variables))

    saliency_q = repeat_saliency(query)
    recon = model.forward(query, saliency_q, training=False)
    return recon[0].numpy(), query[0].numpy()

def visualize_kshot_results(results, class_name):
    os.makedirs("kshot_output", exist_ok=True)

    fig, axs = plt.subplots(2, len(results), figsize=(4 * len(results), 6))
    for i, (k, query_img, recon) in enumerate(results):
        anomaly_map = np.abs(query_img - recon).mean(axis=-1)
        heatmap = cv2.applyColorMap(cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
        input_bgr = cv2.cvtColor((query_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(input_bgr, 0.6, heatmap, 0.4, 0)

        axs[0, i].imshow(recon)
        axs[0, i].set_title(f"{k}-shot\nReconstruction")
        axs[1, i].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axs[1, i].set_title("Anomaly Map")

        axs[0, i].axis('off')
        axs[1, i].axis('off')

        # 儲存圖
        cv2.imwrite(f'kshot_output/{class_name}_{k}shot_recon.jpg', (recon * 255).astype(np.uint8))
        cv2.imwrite(f'kshot_output/{class_name}_{k}shot_overlay.jpg', overlay)

    plt.tight_layout()
    plt.savefig(f'kshot_output/{class_name}_kshot_compare.jpg')
    plt.show()

# === 主程式 ===
if __name__ == "__main__":
    print("📊 K-shot 微調比較開始...")

    class_data = load_all_classes("data", image_size=(64, 64), max_per_class=20)
    class_name = np.random.choice(list(class_data.keys()))
    images = class_data[class_name]

    print(f"📌 測試類別：{class_name}")

    # 固定一張 query 圖
    query_img = images[-1]
    query_tensor = tf.expand_dims(query_img, axis=0)

    results = []
    for k in [1, 3, 5, 10]:
        if len(images) < k + 1:
            print(f"⚠️ {class_name} 類別圖片不足 {k + 1} 張，跳過 {k}-shot")
            continue

        support = preprocess(images[:k])
        query = preprocess(query_tensor)

        model = Metaformer(embed_dim=32)
        model.build(input_shape=[(1, 64, 64, 3), (1, 64, 64, 1)])

        if os.path.exists("metaformer_meta_trained.weights.h5"):
            model.load_weights("metaformer_meta_trained.weights.h5")
        else:
            print("⚠️ 找不到儲存模型，使用未訓練模型")

        recon, qimg = fine_tune_and_infer(model, support, query)
        results.append((k, qimg, recon))

    if results:
        visualize_kshot_results(results, class_name)
        print("✅ 已輸出 k-shot 結果至 kshot_output/")
