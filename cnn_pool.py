import torch
from torch import nn


def pool2D(X: torch.Tensor, pool_size: tuple, mode: str = "max"):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == "max":
                Y[i, j] = X[i : i + p_h, j : j + p_w].max()
            elif mode == "avg":
                Y[i, j] = X[i : i + p_h, j : j + p_w].mean()
    return Y


def main():
    x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    y = pool2D(x, (2, 2), "max")
    print(x, "\n", y)

    pool2d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    x = x.reshape(1, 1, 4, 4)
    print(pool2d(x))


if __name__ == "__main__":
    main()
