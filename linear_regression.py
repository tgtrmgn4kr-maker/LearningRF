import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random as r

import matplotlib.pyplot as plt
import torch


def synthesize_linear_data(w, b, num_examples):
    """Generate y = Xw + b + noise."""
    X = torch.normal(
        0, 1, (num_examples, len(w))
    )  # mu=0, sigma=1 size=(num_examples*len(w))
    y = X @ w + b  # (1000, 2) @ (2,) + (1,) -> (1000, 1)
    y += torch.normal(0, 0.01, y.shape) # y.shape=(1000, 1) noise with mu=0, sigma=0.01
    return X, y.reshape((-1, 1))


def dataloader(features, labels, batch_size):
    """A generator function to yield batches of features and labels."""
    num_examples = len(features)
    indices = list(range(num_examples))
    r.shuffle(indices)  # shuffle the data
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i : min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthesize_linear_data(true_w, true_b, 1000)

print("features:", features[0], "\nlabel:", labels[0])

plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
plt.xlabel("Feature 1")
plt.ylabel("Label")
plt.title("Synthesized Linear Data")
plt.show()

"""
y=Xw+b is a linear regression model, where X is the input features, w is the weights,
b is the bias, and y is the output label.
The function synthesize_linear_data generates synthetic data for linear regression
by creating random features and computing the corresponding labels with some added noise.
The dataloader function is a generator that yields batches of features and labels for training.
"""

batch_size = 10

for X, y in dataloader(features, labels, batch_size):
    print("X:", X, "\ny:", y)
    break

"""
隨機打亂資料，並以小批量的方式讀取資料，這對於訓練模型非常重要，因為它可以幫助模型更好地泛化。
在這個例子中，我們使用了batch_size=10，這意味著每次迭代中，我們將從資料集中隨機選擇10個樣本來訓練模型。
"""
