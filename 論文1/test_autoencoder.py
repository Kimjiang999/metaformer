from metaformer.autoencoder import Autoencoder
import tensorflow as tf

print("✅ 開始測試 Autoencoder...")  # ← 加上這行方便確認程式有執行

model = Autoencoder()
dummy_input = tf.random.normal([1, 256, 256, 3])
reconstructed = model(dummy_input)

print("✅ 測試完成，輸出圖大小：", reconstructed.shape)
