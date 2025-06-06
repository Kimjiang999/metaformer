# plot_auroc_bar.py

import pandas as pd
import matplotlib.pyplot as plt
import os

# 確保 CSV 存在
if not os.path.exists("auroc_result.csv"):
    print("❌ 'auroc_result.csv' not found. Please run eval_auroc_batch.py first.")
    exit()

# 讀取 CSV 結果
df = pd.read_csv("auroc_result.csv")

# 繪製 AUROC 長條圖
plt.figure(figsize=(8, 5))
plt.bar(df["Image"], df["AUROC"], color="royalblue")
plt.ylim(0, 1)
plt.title("AUROC per Image", fontsize=14)
plt.xlabel("Image", fontsize=12)
plt.ylabel("AUROC Score", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

# 儲存與顯示
plt.savefig("auroc_bar_chart.png")
print("✅ Saved AUROC bar chart to 'auroc_bar_chart.png'")
plt.show()
