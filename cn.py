import torch
from torch import nn
from d2l import torch as d2l

class Conv2D(nn.Module):
    def __init__(self, kernel_size) -> None:
        super().__init__()
        self.weight = nn.parameter.Parameter(torch.randn(kernel_size))
        self.bias = nn.parameter.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2D(x, self.weight) + self.bias


def corr2D(X: torch.Tensor, K: torch.Tensor):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y



def main():
    x = torch.arange(9).reshape(3, 3)
    k = torch.arange(4).reshape(2, 2)

    print(corr2D(x, k))  # output: tensor([[19., 25.], [37., 43.]])

    x = torch.ones((6, 8))
    x[:, 2:6] = 0
    k = torch.tensor([[1.0, -1.0]])
    print(x)
    print(x:=corr2D(x, k)) # 如果左右元素相同則輸出0，
    k = k.T
    print(x:=corr2D(x, k)) # 上下元素相同則輸出0
    '''
    簡單的卷積使用範例
    使用卷積檢測左右或上下的元素是否相同
    如果是則輸出0
    '''

    conv2D = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
    

if __name__ == "__main__":
    main()