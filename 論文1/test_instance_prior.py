# test_instance_prior.py
import os
import matplotlib.pyplot as plt
from metaformer.instance_prior import get_instance_prior

print("🟡 測試程式已啟動")
print("📂 目前工作目錄：", os.getcwd())

try:
    image_path = "test_image.jpg"
    print(f"📸 嘗試載入圖片：{image_path}")
    
    saliency = get_instance_prior(image_path)
    print("✅ 成功產生 saliency map，顯示圖片...")

    plt.imshow(saliency, cmap='gray')
    plt.title("Instance-Prior (Saliency Map)")
    plt.axis('off')
    plt.show()

except Exception as e:
    print("❌ 發生錯誤：", e)

cv2.imwrite("output_saliency.jpg", saliency)
