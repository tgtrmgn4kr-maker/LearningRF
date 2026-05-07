import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np
import torch
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
    for i, (ax, img) in enumerate(
        zip(axes, imgs)
    ):  # 使用 zip 函数同时迭代 axes 和 imgs，获取每个子图的 Axes 对象和对应的图像数据。
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.cpu().numpy())
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
    # 如果指定了resize参数，则在transforms列表中添加一个Resize变换，将图像调整为指定的大小。
    if resize:
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def softmax(X: torch.Tensor) -> torch.Tensor:
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制


num_inputs = 784
num_outputs = 10

global w, b
w = torch.normal(
    0, 0.01, size=(num_inputs, num_outputs), requires_grad=True, device=device
)
b = torch.zeros(num_outputs, requires_grad=True, device=device)


def net(X: torch.Tensor) -> torch.Tensor:
    """将输入X重塑为二维张量，第一维是样本数量，第二维是特征数量
    ，然后与权重矩阵w进行矩阵乘法，并加上偏置b，最后通过softmax函数得到预测的概率分布"""

    return softmax(torch.matmul(X.reshape((-1, w.shape[0])), w) + b)


def cross_entropy_loss(
    y_hat: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:  # 交叉熵损失函数
    return -torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:  # 计算预测正确的数量
    if (
        len(y_hat.shape) > 1 and y_hat.shape[1] > 1
    ):  # y大於等於維度2，且第二维的大小大于1，说明y_hat是一个二维张量，且每行包含多个类别的预测概率。
        y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter) -> float:
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
        net.to(device)
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            metric.add(
                accuracy(net(X), y), y.numel()
            )  # 计算当前批次的正确预测数和总预测数，并将它们累加到metric中。
    return metric[0] / metric[1]


def train_epoch(net, train_iter, loss, updater):
    """训练模型一个迭代周期"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(net, torch.nn.Module):
        net.train()  # 将模型设置为训练模式
        net.to(device)
    metric = Accumulator(3)  # 损失总和、正确预测数、预测总数
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)  # 将数据移动到相同的设备
        y_hat = net(X)
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



def train(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(
        xlabel="epoch",
        ylabel="loss/accuracy",
        xlim=[1, num_epochs],
        ylim=[0.3, 0.9],
        legend=["train loss", "train acc", "test acc"],
    )
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
        print(f"epoch {epoch + 1}, loss {train_metrics[0]:.3f}, train acc {train_metrics[1]:.3f}, test acc {test_acc:.3f}")


#  assert train_loss < 0.5, f"训练损失过大: {train_loss}"

#  assert train_acc <= 1 and train_acc > 0.7, f"训练精度过低: {train_acc}"
#  assert test_acc <= 1 and test_acc > 0.7, f"测试精度过低: {test_acc}"


def predict(net, test_iter, n=6):
    """预测标签"""
    for X, y in test_iter:
        break
    X = X.to(device)
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(dim=1).cpu())
    titles = [true + "\n" + pred for true, pred in zip(trues, preds)]
    show_images(
        X[0:n].reshape((n, 28, 28)).cpu(), 1, n, titles=titles[0:n]
    )  # 将前n个测试样本的图像数据重塑为28x28像素的二维张量，并显示出来，标题包含真实标签和预测标签。


def main():
    train_iter, _, mnist_test, mnist_train = load_data_fashion_mnist(
        batch_size=18
    )  # 将图像调整为224x224像素，以适应后续的卷积神经网络输入要求。

    X, y = next(
        iter(train_iter)
    )  # 从训练数据迭代器中获取一个批次的数据，X是图像数据，y是对应的标签。

    show_images(
        X.squeeze(
            1
        ),  # 去掉通道维度，得到形状为(18, 224, 224)的张量，每个图像是224x224像素的灰度图。
        2,  # 行数
        9,  # 列数
        titles=get_fashion_mnist_labels(y),
    )

    plt.show()

    print(
        len(mnist_train), len(mnist_test)
    )  # 输出训练集和测试集的样本数量，分别为60000和10000。

    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        print(f"{timer.stop():.2f} sec")
        break

    batch_size = 256

    train_iter, test_iter, _, _ = load_data_fashion_mnist(batch_size)

    X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    print(
        X.sum(1, keepdim=True)
    )  # 沿着维度1（列）求和，保持维度不变 tensor([[6.],[15.]])))
    print(
        X.sum(0, keepdim=True)
    )  # 沿着维度0（行）求和，保持维度不变 tensor([[5., 7., 9.]]))

    lr = 0.1
    num_epochs = 10

    updater = torch.optim.SGD([w, b], lr=lr)

    train(net, train_iter, test_iter, cross_entropy_loss, num_epochs, updater)
    plt.show()  # 确保图表最后被显示出来

    predict(net, test_iter)
    plt.show()  # 确保图表最后被显示出来


if __name__ == "__main__":
    main()
