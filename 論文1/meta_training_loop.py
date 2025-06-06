# meta_training_loop.py

import tensorflow as tf
import numpy as np
from metaformer.metaformer_model import Metaformer
from utils.data_loader import load_all_classes, sample_meta_task

# === 模型與資料 ===
meta_model = Metaformer(embed_dim=32)
meta_model.build(input_shape=[(1, 64, 64, 3), (1, 64, 64, 1)])
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# 初始化 meta_model 的權重
dummy_input = tf.random.normal([1, 64, 64, 3])
dummy_saliency = tf.random.normal([1, 64, 64, 1])
meta_model.forward(dummy_input, dummy_saliency, training=False)

loss_fn = tf.keras.losses.MeanSquaredError()

# 載入分類圖像資料（每類資料夾）
class_data = load_all_classes("data", image_size=(64, 64), max_per_class=20)

# === MAML Meta-Training 回圈 ===
for step in range(1, 101):  # 你可以改為 1000 或更多
    class_name, support_set, query_set = sample_meta_task(class_data, n_support=5, n_query=5)

    # === Step 1：建立 support 模型副本
    fast_model = Metaformer(embed_dim=32)

    # 先 dummy 運行讓模型 build 起來
    dummy_input = tf.random.normal([1, 64, 64, 3])
    dummy_saliency = tf.random.normal([1, 64, 64, 1])
    fast_model.forward(dummy_input, dummy_saliency, training=False)

    fast_model.set_weights(meta_model.get_weights())  # 複製參數

    # === Step 2：在 support set 上微調 fast_model（inner loop）
    for _ in range(1):  # 可設為多步微調 inner loop
        with tf.GradientTape() as tape:
            saliency = tf.repeat(support_set[..., :1], repeats=3, axis=-1)  # 模擬 saliency
            recon = fast_model.forward(support_set, saliency, training=True)
            loss = loss_fn(support_set, recon)

        grads = tape.gradient(loss, fast_model.trainable_variables)
        k_opt = tf.keras.optimizers.SGD(learning_rate=0.01)
        k_opt.apply_gradients(zip(grads, fast_model.trainable_variables))

    # === Step 3：在 query set 上計算 meta-loss
    saliency_q = tf.repeat(query_set[..., :1], repeats=3, axis=-1)
    recon_q = fast_model.forward(query_set, saliency_q, training=False)
    meta_loss = loss_fn(query_set, recon_q)

    # === Step 4：反向傳遞 meta-loss → 更新 meta_model（outer loop）
    with tf.GradientTape() as meta_tape:
        saliency_s = tf.repeat(support_set[..., :1], repeats=3, axis=-1)
        recon_s = meta_model.forward(support_set, saliency_s, training=True)
        outer_loss = loss_fn(support_set, recon_s)

    meta_grads = meta_tape.gradient(outer_loss, meta_model.trainable_variables)
    optimizer.apply_gradients(zip(meta_grads, meta_model.trainable_variables))

    if step % 10 == 0:
        print(f"Step {step} | Class: {class_name} | Meta-loss: {meta_loss.numpy():.4f}")

# === 結尾
# 儲存訓練後參數
meta_model.save_weights("metaformer_meta_trained.weights.h5")
print("💾 Metaformer 模型已儲存為 metaformer_meta_trained.weights.h5")
