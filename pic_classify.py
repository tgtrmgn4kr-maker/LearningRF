import torch
import torchvision
from torch.utils import data
from torchvision import transforms

trans = transforms.ToTensor()

mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True
)

mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True
)

print(len(mnist_train), len(mnist_test))  # 60000 10000