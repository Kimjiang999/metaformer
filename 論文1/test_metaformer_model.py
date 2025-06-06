# test_metaformer_model.py

import tensorflow as tf
from metaformer.metaformer_model import Metaformer

print("✅ 測試 Metaformer 結構...")

# 建立 Metaformer 模型，降低 embed_dim 以節省記憶體
model = Metaformer(embed_dim=32)

# 減少解析度與通道數
dummy_input = tf.random.normal([1, 64, 64, 3])        # 原始圖像
dummy_saliency = tf.random.normal([1, 64, 64, 1])     # 對應 saliency

# 模型推論
output = model(dummy_input, dummy_saliency)
print("✅ 輸出圖大小：", output.shape)
