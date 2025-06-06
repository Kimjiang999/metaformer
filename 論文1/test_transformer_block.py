# test_transformer_block.py
import tensorflow as tf
from metaformer.transformer_block import TransformerBlock

print("✅ 開始測試 TransformerBlock...")

# 減少解析度與通道數
dummy_input = tf.random.normal([1, 32, 32, 32])  # ✅ 小得多
block = TransformerBlock(embed_dim=32, num_heads=2)

output = block(dummy_input)
print("✅ 測試成功，輸出形狀：", output.shape)
