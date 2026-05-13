import torch
from d2l import torch as d2l
from torch import nn


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

def corr2D_multi_in(X:torch.Tensor, K:torch.Tensor):
        return sum(d2l.corr2d(x, k) for x, k in zip(X, K))

def corr2D_multi_in_out(X:torch.Tensor, K:torch.Tensor):
    return torch.stack([corr2D_multi_in(X, k) for k in K], 0) # type: ignore

def corr2d_multi_in_out_1x1(X:torch.Tensor, K:torch.Tensor):
    """生成多個feature map的卷積結果。"""
    # Input channels, input height, input width
    c_in, h, w = X.shape
    # Output channels, input channels, kernel height, kernel width
    c_out, _, _, _= K.shape
    X = X.reshape((c_in, h * w)) # 圖片的RGB分別轉換成一維矩陣
    K = K.reshape((c_out, c_in))
    Y = K @ X
    return Y.reshape((c_out, h, w))



def main():
    x = torch.arange(9).reshape(3, 3)
    k = torch.arange(4).reshape(2, 2)

    print(corr2D(x, k))  # output: tensor([[19., 25.], [37., 43.]])

    x = torch.ones((6, 8))
    x[:, 2:6] = 0
    k = torch.tensor([[1.0, -1.0]])
    print(x)
    print(y:=corr2D(x, k)) # 如果左右元素相同則輸出0，
    k = k.T
    print(corr2D(x, k)) # 上下元素相同則輸出0
    '''
    簡單的卷積使用範例
    使用卷積檢測左右或上下的元素是否相同
    如果是則輸出0
    '''
    print()

    conv2D = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

    x = x.reshape((1, 1, 6, 8))
    y = y.reshape((1, 1, 6, 7))
    lr = 3e-2
    for i in range(10):
        y_hat = conv2D(x)
        loss_value = ((y_hat - y) ** 2).sum()
        conv2D.zero_grad()
        loss_value.backward()
        if conv2D.weight.data is not None:
            conv2D.weight.data[:] -= lr * conv2D.weight.grad  # type: ignore

        print(f"epoch {i+1}, loss {loss_value.sum():.4f}")

    print(conv2D.weight.data) # tensor([[[[ 0.9704, -1.0079]]]])
    '''
    利用單層CNN檢測左右或上下元素是否相同
    在這個矩陣中
    tensor([[1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.]])
    訓練1*2卷積核
    輸出目標矩陣
    tensor([[0., 1., 0., 0., 0., -1., 0.],
            [0., 1., 0., 0., 0., -1., 0.],
            [0., 1., 0., 0., 0., -1., 0.],
            [0., 1., 0., 0., 0., -1., 0.],
            [0., 1., 0., 0., 0., -1., 0.],
            [0., 1., 0., 0., 0., -1., 0.]])
    '''
    print()

    x = torch.arange(18).reshape(2, 3, 3) # 2個3*3的矩陣
    k = torch.arange(8).reshape(2, 2, 2) # 2個2*2的矩陣
    print(corr2D_multi_in(x, k))  # output: tensor([[268., 296.], [352., 380.]])

    k = torch.stack([k, k+1], 0) # 2個2*2的矩陣
    print(k.shape)
    print(corr2D_multi_in_out(x, k))  # output: tensor([[[268., 296.], [352., 380.]], [[277., 305.], [359., 387.]]])
    '''
    多輸入多輸出卷積使用範例
    利用2個2*2的矩陣做為卷積核
    輸入2個3*3的矩陣
    輸出2個2*2的矩陣
    '''
    print()

    x = torch.normal(0, 1, (3, 3, 3)) # 3 channels, 3x3 images
    k = torch.normal(0, 1, (5, 3, 1, 1)) # 2 output channels, 3 input channels, 1x1 kernel
    print(corr2d_multi_in_out_1x1(x, k)) # output: tensor([[[[-0.0000,  0.0000,  0.0000],


if __name__ == "__main__":
    main()