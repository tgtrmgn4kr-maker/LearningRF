import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms

from tensor_speed import Timer

timer = Timer()


# A enumerable function to get text labels for the Fashion-MNIST dataset
def get_fashion_mnist_labels(labels: torch.Tensor) -> list:
    text_labels = [
        "t-shirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle boot",
    ]
    return [text_labels[int(i)] for i in labels]


def show_images(
    imgs: torch.Tensor, num_rows: int, num_cols: int, titles=None, scale=1.5
):
    """"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    elif isinstance(axes, list):
        axes = [ax for row in axes for ax in row]  # 如果 axes 是嵌套列表，则扁平化
    else:
        axes = [axes]  # 如果 axes 是单个 Axes 对象，则放入列表中
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)

        if ax.axes is not None:
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

        if titles:
            ax.set_title(titles[i])
    return axes


def load_data_fashion_mnist(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load it into memory."""
    trans = []
    if (
        resize
    ):  # 如果指定了resize参数，则在transforms列表中添加一个Resize变换，将图像调整为指定的大小。
        trans.append(transforms.Resize(resize))
    trans.append(
        transforms.ToTensor()
    )  # 将图像转换为PyTorch张量，并将像素值缩放到[0, 1]范围内。

    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=transforms.Compose(trans), download=True
    )

    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=transforms.Compose(trans), download=True
    )

    train_iter = data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, num_workers=4
    )

    test_iter = data.DataLoader(
        mnist_test, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return (
        train_iter,
        test_iter,
        mnist_test,
        mnist_train,
    )

class Accumulator:
    """在n个变量上累加。"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def softmax(X: torch.Tensor) -> torch.Tensor:
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制

num_inputs = 784
num_outputs = 10

w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def net(X: torch.Tensor) -> torch.Tensor:
    return softmax(torch.matmul(X.reshape((-1, w.shape[0])), w) + b)

def cross_entropy(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor: # 交叉熵损失函数
    return -torch.log(y_hat[range(len(y_hat)), y])

def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float: # 计算预测正确的数量
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1: # y大於等於維度2，且第二维的大小大于1，说明y_hat是一个二维张量，且每行包含多个类别的预测概率。
        y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter) -> float:
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期"""
    if isinstance(net, torch.nn.Module):
        net.train()  # 将模型设置为训练模式
    metric = Accumulator(3)  # 损失总和、正确预测数、预测总数
    for X, y in train_iter:
        y_hat = net(X)
        loss_value = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            loss_value.mean().backward()
            updater.step()
        else:
            loss_value.sum().backward()
            updater(X.shape[0])
        metric.add(float(loss_value.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

def main():
    train_iter, _, mnist_test, mnist_train = load_data_fashion_mnist(
        batch_size=18, resize=224
    )  # 将图像调整为224x224像素，以适应后续的卷积神经网络输入要求。

    X, y = next(
        iter(train_iter)
    )  # 从训练数据迭代器中获取一个批次的数据，X是图像数据，y是对应的标签。

    show_images(
        X.reshape(
            18, 224, 224
        ),  # 18张图像，每张图像是224x224像素的单通道图像，因此将X重塑为(18, 224, 224)的形状。
        2,  # 行数
        9,  # 列数
        titles=get_fashion_mnist_labels(y),
    )

    plt.show()

    print(len(mnist_train), len(mnist_test)) # 输出训练集和测试集的样本数量，分别为60000和10000。

    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        print(f"{timer.stop():.2f} sec")
        break

    # 下面的代码展示了如何使用softmax函数将模型的输出转换为概率分布。

    batch_size = 256

    train_iter, test_iter, _, _ = load_data_fashion_mnist(batch_size)

    X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    print(
        X.sum(1, keepdim=True)
    )  # 沿着维度1（列）求和，保持维度不变 tensor([[6.],[15.]])
    print(
        X.sum(0, keepdim=True)
    )  # 沿着维度0（行）求和，保持维度不变 tensor([[5., 7., 9.]])


if __name__ == "__main__":
    main()
