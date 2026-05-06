# =========================
# build_graph.py
# =========================

import pandas as pd
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# =========================
# 1️⃣ 读取社区数据
# =========================
data_path = r"D:\大学\code\data_analysis\community_data.csv"

community_df = pd.read_csv(data_path)

print("数据读取成功:", community_df.shape)

# =========================
# 2️⃣ 选择特征
# =========================
feature_cols = ['ba001', 'ba002', 'da001', 'da002', 'ca001', 'gb001']

X = community_df[feature_cols].values

# =========================
# 3️⃣ 标准化（非常重要）
# =========================
scaler = StandardScaler()
X = scaler.fit_transform(X)

print("特征标准化完成:", X.shape)

# =========================
# 4️⃣ KNN建图（优化版）
# =========================
k = 5  # 可以调：3 / 5 / 10

knn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
knn.fit(X)

distances, indices = knn.kneighbors(X)

# =========================
# 5️⃣ 构建 edge_index + edge_weight
# =========================
edge_list = []
edge_weight = []

for i in range(len(indices)):
    for idx, j in enumerate(indices[i][1:]):  # 跳过自己
        edge_list.append([i, j])
        
        # 👉 用距离转权重（越近越大）
        dist = distances[i][idx + 1]
        weight = 1 / (1 + dist)
        edge_weight.append(weight)

# 转 tensor
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
edge_weight = torch.tensor(edge_weight, dtype=torch.float)

print("edge_index shape:", edge_index.shape)
print("edge_weight shape:", edge_weight.shape)

# =========================
# 6️⃣ 转无向图（关键优化）
# =========================
edge_index_rev = edge_index[[1, 0]]
edge_index = torch.cat([edge_index, edge_index_rev], dim=1)

edge_weight = torch.cat([edge_weight, edge_weight], dim=0)

print("无向图 edge_index:", edge_index.shape)

# =========================
# 7️⃣ 保存结果
# =========================
torch.save(edge_index, "edge_index.pt")
torch.save(edge_weight, "edge_weight.pt")

print("✅ 图构建完成，已保存！")

# =========================
# 8️⃣ 简单检查
# =========================
num_nodes = X.shape[0]
num_edges = edge_index.shape[1]

print(f"节点数: {num_nodes}")
print(f"边数: {num_edges}")
print(f"平均每个节点连接: {num_edges / num_nodes:.2f}")