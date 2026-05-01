import matplotlib.pyplot as plt
import torch
from torch import nn

from softmax_regression import load_data_fashion_mnist, train

batch_size = 256
num_epochs = 10
lr = 0.01
num_inputs, num_outputs, num_hidden = 784, 10, 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

W1 = nn.Parameter(
    torch.randn((num_inputs, num_hidden), device=device) * 0.01, requires_grad=True
) # 784输入特征，256隐藏单元的权重矩阵，初始化为均值为0，标准差为0.01的正态分布，并且需要计算梯度
b1 = nn.Parameter(torch.zeros((num_hidden,), device=device), requires_grad=True)
W2 = nn.Parameter(
    torch.randn((num_hidden, num_outputs), device=device) * 0.01, requires_grad=True
)
b2 = nn.Parameter(torch.zeros((num_outputs,), device=device), requires_grad=True)

params = [W1, b1, W2, b2]
updater = torch.optim.SGD(params, lr=lr)


def relu(x):
    return torch.max(torch.zeros_like(x), x)


def net(X):
    X = X.to(device)
    X = X.reshape(-1, num_inputs)
    H = relu(X @ W1 + b1)
    return H @ W2 + b2


def main():

    train_iter, test_iter, _, _ = load_data_fashion_mnist(batch_size)

    loss = nn.CrossEntropyLoss(reduction="none")
    train(net, train_iter, test_iter, loss, num_epochs, updater)
    plt.show()


if __name__ == "__main__":
    main()
