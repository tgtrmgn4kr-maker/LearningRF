import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


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


class Animator:
    """在动画中绘制数据。"""

    def __init__(
        self,
        xlabel=None,
        ylabel=None,
        legend=None,
        xlim=None,
        ylim=None,
        xscale="linear",
        yscale="linear",
        fmts=("-r", "m--", "g-.", "b:"),
        nrows=1,
        ncols=1,
        figsize=(3.5, 2.5),
    ):
        # 增加了一个参数fmts，表示不同数据序列的线条格式，默认为("-r", "m--", "g-.", "b:")。
        if legend is None:
            legend = []
        plt.rcParams["figure.figsize"] = figsize
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if isinstance(self.axes, np.ndarray):
            self.axes = self.axes.flatten()[0]
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_xscale(xscale)
        self.axes.set_yscale(yscale)
        self.axes.set_xlim(xlim)
        self.axes.set_ylim(ylim)
        self.legend = legend
        self.fmts = fmts

        self.config_axes = (
            lambda: self.axes.set_xlabel(xlabel)
            or self.axes.set_ylabel(ylabel)
            or self.axes.set_xscale(xscale)
            or self.axes.set_yscale(yscale)
            or self.axes.set_xlim(xlim)
            or self.axes.set_ylim(ylim)
        )

        self.X, self.Y = None, None
        plt.show(block=False)

    def add(self, x, y):
        # 在add方法中，我们使用了zip函数将x、y和fmts三个列表打包在一起，这样就可以同时迭代它们。
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if self.X is None:
            self.X = [[] for _ in range(n)]
        if self.Y is None:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)

        self.axes.cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes.plot(x, y, fmt)
        if self.legend:
            self.axes.legend(self.legend)

        self.config_axes()
        self.fig.canvas.draw()
        plt.pause(0.001)


def evaluate_loss(net, data_iter, loss):
    """Evaluate the loss of a model on the given dataset."""
    metric = Accumulator(2)  # Sum of losses, no. of examples
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 获取设备
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)  # 将数据移动到相同的设备
        output = net(X)
        y = y.reshape(output.shape)  # 确保 y 的形状与 output 的形状一致
        loss_value = loss(output, y)
        metric.add(loss_value.sum().item(), y.numel())
    return metric[0] / metric[1]


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个数据迭代器。"""
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)


def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:  # 计算预测正确的数量
    if (
        len(y_hat.shape) > 1 and y_hat.shape[1] > 1
    ):  # y大於等於維度2，且第二维的大小大于1，说明y_hat是一个二维张量，且每行包含多个类别的预测概率。
        y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def train_epoch(net, train_iter, loss, updater):
    """训练模型一个迭代周期"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(net, torch.nn.Module):
        net.train()  # 将模型设置为训练模式
        net.to(device)
    metric = Accumulator(3)  # 损失总和、正确预测数、预测总数
    for X, y in train_iter:  # features, labels
        X, y = X.to(device), y.to(device)  # 将数据移动到相同的设备
        y_hat = net(X)  # net built with features
        y = y.reshape(y_hat.shape)  # 确保 y 的形状与 output 的形状一致
        loss_value = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            loss_value.mean().backward()
            updater.step()
        else:
            loss_value.sum().backward()
            updater(X.shape[0])
        metric.add(loss_value.sum().item(), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]
    # metric[0]是损失总和，metric[2]是预测总数，metric[1]是正确预测数，所以返回的第一个值是平均损失，第二个值是平均精度。


def train(
    train_features, test_features, train_labels, test_labels, num_epochs=1000, lr=0.005
):
    """Train a polynomial regression model."""
    loss = torch.nn.MSELoss()
    input_shape = train_features.shape[-1]
    # Single layer model
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))  # No bias

    batch_size = min(10, train_labels.shape[0])
    train_iter = load_array((train_features, train_labels), batch_size)
    test_iter = load_array((test_features, test_labels), batch_size, is_train=False)

    # 梯度下降，对于高阶多项式，可能需要调整学习率
    trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0)

    animator = Animator(
        xlabel="epoch",
        ylabel="loss",
        xlim=[1, num_epochs],
        ylim=[0.001, 1],
        yscale="log",
        legend=["train", "test"],
    )

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(net, train_iter, loss, trainer)
        if (epoch + 1) % 20 == 0:
            animator.add(
                epoch + 1,
                (
                    evaluate_loss(net, train_iter, loss),
                    evaluate_loss(net, test_iter, loss),
                ),
            )

    weight = net[0].weight.detach().cpu().numpy()  # type: ignore
    print(f"Weight for degree {input_shape}:", weight)
    print(f"Final train loss: {evaluate_loss(net, train_iter, loss):.6f}")
    print(f"Final test loss: {evaluate_loss(net, test_iter, loss):.6f}")
    print("-" * 50)

    return evaluate_loss(net, train_iter, loss), evaluate_loss(net, test_iter, loss)
