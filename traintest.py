import torch
import torch.nn as nn
import torch.optim as optim

# 随机数据（假装是训练数据）
x = torch.randn(100, 1)
y = 2 * x + 1

# 定义模型
model = nn.Linear(1, 1)

# 损失函数
criterion = nn.MSELoss()

# 优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练
for epoch in range(100):
    pred = model(x)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"epoch {epoch}, loss {loss.item()}")