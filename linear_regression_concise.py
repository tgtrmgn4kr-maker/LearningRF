import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random as r

import torch
from torch import nn


def synthesize_linear_data(w, b, num_examples):
    """Generate y = Xw + b + noise."""
    X = torch.normal(
        0, 1, (num_examples, len(w))
    )  # mu=0, sigma=1 size=(num_examples*len(w))
    y = X @ w + b  # (1000, 2) @ (2,) + (1,) -> (1000, 1)
    y += torch.normal(0, 0.01, y.shape)  # y.shape=(1000, 1) noise with mu=0, sigma=0.01
    return X, y.reshape((-1, 1))


def dataloader(features, labels, batch_size):
    """A generator function to yield batches of features and labels."""
    num_examples = len(features)
    indices = list(range(num_examples))
    r.shuffle(indices)  # shuffle the data
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i : min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator."""
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthesize_linear_data(true_w, true_b, 1000)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

print(next(iter(data_iter)))  # 一組 features 和 labels 的小批量数据

net = nn.Sequential(nn.Linear(2, 1))
"""
Sequential是一個容器，將多個層組合在一起，按照順序執行。
在這裡，我們使用了一個線性層nn.Linear(2, 1)，輸入特徵的維度是2，輸出特徵的維度是1。
"""

print(net[0].weight.data)  # 隨機初始化的權重
print(net[0].bias.data)  # 隨機初始化的偏置

loss = nn.MSELoss()  # 均方誤差損失函數
trainer = torch.optim.SGD(net.parameters(), lr=0.03)  # 隨機梯度下降優化器，學習率為0.03

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        loss_value = loss(net(X), y)  # 計算損失
        trainer.zero_grad()  # 清空梯度
        loss_value.backward()  # 反向傳播計算梯度
        trainer.step()  # 更新參數

    with torch.no_grad():
        train_l = loss(net(features), labels)
        print(f"epoch {epoch + 1}, loss {train_l:f}")
