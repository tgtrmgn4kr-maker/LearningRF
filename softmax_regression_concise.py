

import matplotlib.pyplot as plt
import torch
from torch import nn

from softmax_regression import load_data_fashion_mnist, train


def init_weights(m: nn.Module):
    if type(m) is nn.Linear:
        nn.init.normal_(
            mean=0, std=0.01, tensor=m.weight
        )  # 初始化权重为均值为0，标准差为0.01的正态分布





def main():

    batch_size = 256

    # 定义一个包含两个层的神经网络：一个Flatten层将输入数据展平为一维向量，另一个Linear层将展平后的输入映射到10个输出类别。
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    net.apply(init_weights)  # 对网络中的每个模块应用init_weights函数，初始化权重

    loss = nn.CrossEntropyLoss(reduction='none')  # 定义交叉熵损失函数

    trainer = torch.optim.SGD(
        net.parameters(), lr=0.01
    )  # 定义随机梯度下降优化器，学习率为0.01

    train_iter, test_iter, _, _ = load_data_fashion_mnist(
        batch_size
    )  # 加载Fashion-MNIST数据集，返回训练和测试数据的迭代器，以及训练和测试数据的样本数量

    num_epochs = 10
    train(net, train_iter, test_iter, loss, num_epochs, trainer)
    plt.show()  # 确保图表最后被显示出来


if __name__ == "__main__":
    main()
