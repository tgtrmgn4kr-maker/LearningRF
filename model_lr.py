import torch
from torch import nn
from torch.utils import data


def synthesize_linear_data(w: torch.Tensor, b: float, num_examples: int):
    """Generate y = Xw + b + noise."""
    X = torch.normal(
        0, 1, (num_examples, len(w))
    )  # mu=0, sigma=1 size=(num_examples*len(w))
    y = X @ w + b  # (1000, 2) @ (2,) + (1,) -> (1000, 1)
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def load_array(data_arrays: tuple, batch_size: int, is_train: bool = True):
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


true_w = torch.tensor([2, 3.4])
true_b = 4.2

features, labels = synthesize_linear_data(
    true_w, true_b, 1000
)  # 生成1000個在二維空間中的線性數據，特徵維度為2，標籤為1
batch_size = 5

# 使用load_array函數將生成的特徵和標籤數據轉換為PyTorch的DataLoader對象，這樣可以方便地進行批量數據加載和迭代訓練模型
data_iter = load_array((features, labels), batch_size)

print(next(iter(data_iter)))
"""
從數據迭代器中獲取第一個批次的特徵和標籤，並打印出來以檢查數據是否正確加載。
這裡使用了iter()函數將DataLoader對象轉換為迭代器，然後使用next()函數獲取第一個批次的數據。
"""

net = nn.Sequential(nn.Linear(2, 1))  # 定義一個線性模型，輸入特徵維度為2，輸出維度為1

loss = nn.MSELoss()  # 定義均方誤差損失函數

trainer = torch.optim.SGD(
    net.parameters(), lr=0.03
)  # 定義優化器，使用隨機梯度下降算法，學習率為0.03

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        loss_value = loss(net(X), y)  # 計算模型預測值與真實標籤之間的損失
        trainer.zero_grad()  # 清除之前的梯度
        loss_value.backward()  # 反向傳播計算梯度
        trainer.step()  # 更新模型參數

    with torch.no_grad():
        train_l = loss(
            net(features), labels
        )  # 在整個訓練數據上計算損失，以評估模型的性能
        print(f"epoch {epoch + 1}, loss {train_l:f}")

w = net[0].weight.data
b = net[0].bias.data
print(f"Desired w: {true_w}, b: {true_b}")
print(f"Estimated w: {w}, Estimated b: {b}")
"""
epoch 1, loss 0.000101
epoch 2, loss 0.000105
epoch 3, loss 0.000102

Desired w: tensor([2.0000, 3.4000]), b: 4.2
Estimated w: tensor([[1.9991, 3.3993]]), Estimated b: tensor([4.2003])
"""
