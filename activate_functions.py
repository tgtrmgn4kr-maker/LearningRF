import matplotlib.pyplot as plt
import torch

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)  # 从-8到8，步长为0.1的张量，并且需要计算梯度
y_relu = torch.relu(x)
y_relu.backward(torch.ones_like(x))  # 计算y对x的梯度

# 绘制ReLU函数及其梯度
plt.plot(x.detach().numpy(), y_relu.detach().numpy(), label="ReLU")
plt.xlabel("x")
plt.ylabel("ReLU(x)")
plt.title("ReLU Activation Function")
plt.legend()
plt.show()

# 绘制Sigmoid函数及其梯度
y_sigmoid = torch.sigmoid(x)
plt.plot(x.detach().numpy(), y_sigmoid.detach().numpy(), label="Sigmoid")
plt.xlabel("x")
plt.ylabel("Sigmoid(x)")
plt.title("Sigmoid Activation Function")
plt.legend()
plt.show()

if x.grad is not None:
    x.grad.zero_()

    y_sigmoid.backward(torch.ones_like(x))  # 计算y对x的梯度
    plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label="Gradient of Sigmoid")
    plt.xlabel("x")
    plt.ylabel("Gradient of Sigmoid(x)")
    plt.title("Gradient of Sigmoid Activation Function")
    plt.legend()
    plt.show()

# 绘制Tanh函数及其梯度
y_tanh = torch.tanh(x)
plt.plot(x.detach().numpy(), y_tanh.detach().numpy(), label="Tanh")
plt.xlabel("x")
plt.ylabel("Tanh(x)")
plt.title("Tanh Activation Function")
plt.legend()
plt.show()

if x.grad is not None:
    x.grad.zero_()

    y = torch.tanh(x)
    y.backward(torch.ones_like(x))
    plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label="Gradient of Tanh")
    plt.xlabel("x")
    plt.ylabel("Gradient of Tanh(x)")
    plt.title("Gradient of Tanh Activation Function")
    plt.legend()
    plt.show()
