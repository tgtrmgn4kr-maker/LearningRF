import torch
from torch import nn

from softmax_regression import Accumulator, Animator, accuracy, load_data_fashion_mnist
from tensor_speed import Timer


def evaluate_accuracy(net, data_iter):
    if isinstance(net, nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) is nn.Linear or type(m) is nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(
        xlabel="epoch",
        xlim=[1, num_epochs],
        ylim=[0.3, 0.9],
        legend=["train loss", "train acc", "test acc"],
    )
    timer, num_batches = Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = Accumulator(3)  # 训练损失之和、训练准确度之和、样本数
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            loss_value = loss(y_hat, y)
            loss_value.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(
                    loss_value.item() * X.shape[0], accuracy(y_hat, y), X.shape[0]
                )
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f"loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}")


def main():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
        # (1, 28, 28) -> (6, 28, 28)
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        # (6, 28, 28) -> (6, 14, 14)
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
        # (6, 14, 14) -> (16, 10, 10)
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        # (16, 10, 10) -> (16, 5, 5)
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        nn.Sigmoid(),
        nn.Linear(84, 10),
    )

    x = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)

    for layer in net:
        x = layer(x)
        print(layer.__class__.__name__, "output shape:\t", x.shape)

    batch_size = 256
    train_iter, test_iter, _, _ = load_data_fashion_mnist(batch_size=batch_size)

    lr, num_epochs = 0.001, 10
    train_(net, train_iter, test_iter, num_epochs, lr, device)


if __name__ == "__main__":
    main()
