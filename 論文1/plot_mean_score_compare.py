# plot_mean_score_compare.py

import pandas as pd
import matplotlib.pyplot as plt
import os

# 檢查資料是否存在
if not os.path.exists("auroc_result.csv"):
    print("❌ 'auroc_result.csv' not found. Please run eval_auroc_batch.py first.")
    exit()

# 讀取 CSV
try:
    df = pd.read_csv("auroc_result.csv")
except Exception as e:
    print("❌ Failed to load CSV:", e)
    exit()

# 判斷是否有 before/after 分組
if not df["Image"].str.contains("before").any() or not df["Image"].str.contains("after").any():
    print("⚠️ No 'before' and 'after' labels found in 'Image' column. Please ensure your image names follow '*_before' and '*_after' pattern.")
    exit()

# 依據名稱拆分類別與狀態
df["Category"] = df["Image"].str.replace("_before", "").str.replace("_after", "")
df["Status"] = df["Image"].apply(lambda x: "Before" if "before" in x else ("After" if "after" in x else "Unknown"))

# 繪製
plt.figure(figsize=(9, 5))
for category in df["Category"].unique():
    sub = df[df["Category"] == category]
    if len(sub) == 2:
        before = sub[sub["Status"] == "Before"]["MeanScore"].values[0]
        after = sub[sub["Status"] == "After"]["MeanScore"].values[0]
        plt.plot(["Before", "After"], [before, after], marker='o', label=category)

plt.title("Mean Anomaly Score: Before vs After Fine-tuning", fontsize=14)
plt.ylabel("Mean Anomaly Score", fontsize=12)
plt.xlabel("Stage", fontsize=12)
plt.legend(title="Image Category")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("mean_score_comparison.png")
print("✅ Saved mean score comparison chart to 'mean_score_comparison.png'")
plt.show()
