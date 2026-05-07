# =========================
# compare_result.py
# GNN预测结果可视化
# =========================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# =========================
# 1️⃣ 读取预测结果
# =========================
result_path = r"D:\大学\code\GNN\predict_result.csv"

df = pd.read_csv(result_path)

y_true = df['true_label']
y_pred = df['pred_label']

print(df.head())

# =========================
# 2️⃣ 分类准确率
# =========================
acc = accuracy_score(y_true, y_pred)

print("\n分类准确率：", acc)

# =========================
# 3️⃣ 分类报告
# =========================
print("\n分类报告：")
print(classification_report(y_true, y_pred))

# =========================
# 4️⃣ 真实值 vs 预测值
# =========================
plt.figure(figsize=(12, 6))

plt.plot(
    y_true.values[:80],
    label='True Label',
    linewidth=2
)

plt.plot(
    y_pred.values[:80],
    label='Pred Label',
    linewidth=2
)

plt.xlabel("Community Node")
plt.ylabel("Need Level")

plt.title("True vs Predicted Labels")

plt.legend()

plt.grid(True)

# 保存
plt.savefig(
    r"D:\大学\code\GNN\predict_compare.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# =========================
# 5️⃣ 混淆矩阵
# =========================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.colorbar()

plt.xlabel("Predicted")
plt.ylabel("True")

plt.title("Confusion Matrix")

# 添加数字
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i, j],
            ha='center',
            va='center'
        )

plt.savefig(
    r"D:\大学\code\GNN\confusion_matrix.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print("\n图像保存完成")