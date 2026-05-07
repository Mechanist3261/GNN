# =========================
# train_gnn.py
# =========================

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# =========================
# 1️⃣ 读取数据
# =========================
community_df = pd.read_csv(r"D:\大学\code\data_analysis\community_data.csv")

edge_index = torch.load("edge_index.pt")
edge_weight = torch.load("edge_weight.pt")

print("数据读取完成")

# =========================
# 2️⃣ 构建输入
# =========================
feature_cols = ['ba001', 'ba002', 'da001', 'da002', 'ca001', 'gb001']

x = torch.tensor(community_df[feature_cols].values, dtype=torch.float)

y_reg = torch.tensor(community_df['need_score'].values, dtype=torch.float)
y_cls = torch.tensor(community_df['need_level'].values, dtype=torch.long)

data = Data(x=x, edge_index=edge_index)
data.edge_attr = edge_weight
data.y_reg = y_reg
data.y_cls = y_cls

print(data)

# =========================
# 3️⃣ 定义模型（多任务）
# =========================
class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=32):
        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # 👉 回归头
        self.reg_head = nn.Linear(hidden_dim, 1)

        # 👉 分类头（5类）
        self.cls_head = nn.Linear(hidden_dim, 5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        reg_out = self.reg_head(x).squeeze()
        cls_out = self.cls_head(x)

        return reg_out, cls_out


# =========================
# 4️⃣ 初始化
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GNN(in_dim=x.shape[1]).to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# =========================
# 5️⃣ 训练
# =========================
def train():
    model.train()
    optimizer.zero_grad()

    reg_out, cls_out = model(data.x, data.edge_index)

    # 👉 回归损失
    loss_reg = F.mse_loss(reg_out, data.y_reg)

    # 👉 分类损失
    loss_cls = F.cross_entropy(cls_out, data.y_cls)

    # 👉 总损失（可调权重）
    loss = loss_reg + loss_cls

    loss.backward()
    optimizer.step()

    return loss.item(), loss_reg.item(), loss_cls.item()


# =========================
# 6️⃣ 训练循环
# =========================
for epoch in range(1, 201):
    loss, loss_reg, loss_cls = train()

    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | 总loss: {loss:.4f} | 回归: {loss_reg:.4f} | 分类: {loss_cls:.4f}")


# =========================
# 7️⃣ 简单评估
# =========================
model.eval()

with torch.no_grad():
    reg_out, cls_out = model(data.x, data.edge_index)

    pred_cls = cls_out.argmax(dim=1)

    # =========================
    # 保存预测结果
    # =========================
    result_df = pd.DataFrame({
        'true_label': y_cls.cpu().numpy(),
        'pred_label': pred_cls.cpu().numpy()
    })

    result_df.to_csv(
        r"D:\大学\code\GNN\predict_result.csv",
        index=False
    )

    print("预测结果已保存")

    acc = (pred_cls == data.y_cls).sum().item() / len(data.y_cls)

    print("\n分类准确率:", acc)

    print("\n预测示例：")
    print(pred_cls[:10].cpu().numpy())