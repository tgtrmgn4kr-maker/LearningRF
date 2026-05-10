import torch
from d2l import torch as d2l
from torch import nn
from softmax_regression import load_data_fashion_mnist

def dropout_layer(X: torch.Tensor, dropout: float):
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)

n_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter, _, _ = load_data_fashion_mnist(batch_size)

def init_weights(m):
    if type(m) is nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

def main():
    train_iter, test_iter, _, _ = load_data_fashion_mnist(batch_size)
    # 添加展平操作
    train_iter = [(X.view(X.shape[0], -1), y) for X, y in train_iter]
    test_iter = [(X.view(X.shape[0], -1), y) for X, y in test_iter]

    net = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 10)
    )
    net.apply(init_weights)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    d2l.train_ch3(net, train_iter, test_iter, loss, n_epochs, trainer)
    d2l.plt.show()



if __name__ == '__main__':
    main()