# =========================
# loss_curve.py
# GNN损失曲线可视化
# =========================

import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1️⃣ 读取loss数据
# =========================
loss_path = r"D:\大学\code\GNN\loss_history.csv"

loss_df = pd.read_csv(loss_path)

print(loss_df.head())

# =========================
# 2️⃣ 设置画布
# =========================
plt.figure(figsize=(10, 6))

# =========================
# 3️⃣ 绘制曲线
# =========================
plt.plot(
    loss_df['epoch'],
    loss_df['total_loss'],
    label='Total Loss',
    linewidth=2
)

plt.plot(
    loss_df['epoch'],
    loss_df['reg_loss'],
    label='Regression Loss',
    linewidth=2
)

plt.plot(
    loss_df['epoch'],
    loss_df['cls_loss'],
    label='Classification Loss',
    linewidth=2
)

# =========================
# 4️⃣ 图形美化
# =========================
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)

plt.title("GNN Training Loss Curve", fontsize=16)

plt.legend()

plt.grid(True)

# =========================
# 5️⃣ 保存图片
# =========================
save_path = r"D:\大学\code\GNN\loss_curve.png"

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches='tight'
)

print("损失曲线已保存：", save_path)

# =========================
# 6️⃣ 显示图像
# =========================
plt.show()