import torch

x = torch.arange(4.0, requires_grad=True)  # leaf tensor 使用者建立的tensor 計算圖的起點
"""
PyTorch 每個 tensor 主要有幾個重要屬性：
- tensor.data      # tensor 本身的數值
- tensor.grad      # 梯度(只有 leaf tensor 才有)，在反向傳播後存儲梯度值
- tensor.grad_fn   # 這個 tensor 的計算來源
"""
y = 2 * torch.dot(x, x)  # non-leaf tensor 不是使用者建立的tensor 計算圖的中間節點
# y是x和x的內積，然後乘以2

print(y)  # tensor(28., grad_fn=<MulBackward0>)

y.backward()
# 反向傳播，計算y對x的梯度，將結果存儲在x.grad中
# y = 2 * ( x*x ) = 2 * ( x^2 ) => dy/dx = 4 * x

print(x.grad)  # 打印x的梯度，應該是4 * x

print(x.grad == 4 * x)  # Check if the gradient is correct

if x.grad is not None:
    x.grad.zero_()  # 將x的梯度清零，以便下一次計算
"""
y=x和x的內積，然後乘以2，所以y=2*(x^2)，對x求導得到dy/dx=4*x，因此x.grad應該等於4*x。
"""

y = x.sum()  # y = x1 + x2 + x3 + x4 => dy/dx = 1 (對每個元素求導都是1)
y.backward()  # 反向傳播，計算y對x的梯度，將結果存儲在x.grad中
print(x.grad)  # tensor([1., 1., 1., 1.])
if x.grad is not None:
    x.grad.zero_()

"""
y=x dy/dx=1，所以x.grad應該是全1的tensor。
"""

y = x * x  # x * x == torch.dot(x, x) == x^2

y.sum().backward()  # y.sum() = x1^2 + x2^2 + x3^2 + x4^2 => dy/dx = 2*x
print(x.grad)  # tensor([ 0.,  2.,  4.,  6.])
if x.grad is not None:
    x.grad.zero_()
"""
和第一個計算差不多
"""

y = x * x  # y是x和x的內積 non-leaf tensor
u = y.detach()  # u是y的副本，但不會追蹤梯度，是一個non-leaf tensor
z = u * x
z.sum().backward()  # z.sum() = u1*x1 + u2*x2 + u3*x3 + u4*x4 => dz/dx = u
print(x.grad)  # tensor([0., 1., 4., 9.])
print(x.grad == u)  # tensor([True, True, True, True])
if x.grad is not None:
    x.grad.zero_()

y.sum().backward()  # y.sum() = x1^2 + x2^2 + x3^2 + x4^2 => dy/dx = 2*x
print(x.grad==2*x)  # tensor([True, True, True, True])
if x.grad is not None:
    x.grad.zero_()
"""
因為u是y的副本，但不會追蹤梯度，所以z.sum().backward()是z對x的導數就是u，
而不是y對x的導數，然後儲存至x.grad
"""

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        return b
    else:
        return b * 100

a = torch.randn(size=(), requires_grad=True)  # a是一個純量tensor(只有值沒有維度)
print(a)  # 打印a的值
b = f(a)  # b是f(a)的結果，是一個non-leaf tensor

b.backward()

print(a.grad == b/a)  # 因為b是a的函數，對a求導得到db/da = b/a，所以a.grad應該等於b/a

if a.grad is not None:
    a.grad.zero_()
