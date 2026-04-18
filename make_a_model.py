import random as r

import torch

'''
深度學習框架的底層示範
=========================

線性回歸模型的訓練過程包括以下步驟：
1. 定義模型：線性回歸模型的形式為y = Xw + b，其中X是輸入特徵，w是權重，b是偏置，y是輸出標籤。
2. 定義損失函數：常用的損失函數是均方誤差（MSE），計算預測值與真實值之間的差距。
3. 定義優化算法：使用小批量隨機梯度下降（SGD）來更新模型參數，通過計算損失函數對權重和偏置的梯度來進行更新。
4. 訓練模型：迭代多個epoch，每個epoch中使用dataloader讀取小批量的數據，計算損失，反向傳播計算梯度，並使用優化算法更新參數。
5. 評估模型：在訓練過程中，可以計算整個資料集的損失來評估模型的性能。
在這個例子中，我們使用了synthetic data來訓練線性回歸模型，並且通過多次迭代來優化模型參數，使得模型能夠更好地擬合數據。
'''

def linear_regression(X, w, b):
    """The linear regression model."""
    return torch.matmul(X, w) + b


def squared_loss(y_hat: torch.Tensor, y: torch.Tensor):
    """Squared loss."""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params: list, lr: float, batch_size: int):
    """Minibatch stochastic gradient descent."""
    """小批量隨機梯度下降，更新模型參數"""
    with torch.no_grad():  # 這裡不需要計算梯度(已經計算完畢)，所以使用torch.no_grad()來禁用梯度計算，這樣可以節省內存和計算資源
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def dataloader(features: torch.Tensor, labels: torch.Tensor, batch_size: int):
    """A generator function to yield batches of features and labels."""
    num_examples = len(features)
    indices = list(range(num_examples))
    r.shuffle(indices)  # shuffle the data
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i : min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def synthesize_linear_data(w: torch.Tensor, b: float, num_examples: int):
    """Generate y = Xw + b + noise."""
    X = torch.normal(
        0, 1, (num_examples, len(w))
    )  # mu=0, sigma=1 size=(num_examples*len(w))
    y = X @ w + b  # (1000, 2) @ (2,) + (1,) -> (1000, 1)
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthesize_linear_data(true_w, true_b, 1000)
batch_size = 16

print("features:", features[0], "\nlabel:", labels[0])

# 超參數設定
learning_rate = 0.03
num_epochs = 3

# model parameters
net = linear_regression
loss = squared_loss

# 需要訓練的參數，這裡我們使用requires_grad=True來告訴PyTorch需要對這些參數計算梯度
weight = torch.normal(
    0, 0.01, size=(2,), requires_grad=True
)  # 這裡的size=(2,)是因為我們有兩個特徵，所以權重向量的大小應該是2
bias = torch.zeros(1, requires_grad=True)

for epoch in range(num_epochs):
    for X, y in dataloader(features, labels, batch_size):  # 讀取一個小批量
        loss_value = loss(net(X, weight, bias), y)  # compute loss of a batch
        loss_value.sum().backward()  # 計算loss對weight和bias的梯度
        # ∂loss/∂weight => weight.grad
        # ∂loss/∂bias => bias.grad
        sgd(
            [weight, bias], learning_rate, batch_size
        )  # update parameters using their gradient

    with torch.no_grad():  # 計算整個資料集的loss，這裡不需要計算梯度，所以使用torch.no_grad()來禁用梯度計算
        train_l = loss(net(features, weight, bias), labels)
        print(f"epoch {epoch + 1}, loss {float(train_l.mean()):f}")
