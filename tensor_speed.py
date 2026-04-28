import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch


class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()


def normal(x, mu, sigma):
    p = 1 / (sigma * math.sqrt(2 * math.pi))
    return p * np.exp(-0.5 / sigma**2 * (x - mu) ** 2)

def main():
    n = 100000

    a = torch.zeros(n)
    b = torch.zeros(n)
    c = torch.zeros(n)
    t = Timer()
    for i in range(n):
        c[i] = a[i] + b[i]
    print(f"Time: {t.stop():.5f} seconds")  # 1.46160 seconds

    t.start()
    c = a + b
    print(f"Time: {t.stop():.5f} seconds")  # 0.00097 seconds
    """
    Pytorch's vectorized operations are much faster than Python loops,
    especially for large tensors.
    In this example, the loop takes significantly more time than the vectorized addition.
    """

    x = np.arange(-7, 7, 0.1)

    params = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)]

    plt.figure(figsize=(6, 4.5))
    for mu, sigma in params:
        plt.plot(x, normal(x, mu, sigma), label=f"mu={mu}, sigma={sigma}")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.title("Normal Distributions")
    plt.show()
    """
    畫出不同均值（mu）和標準差（sigma）的正態分布曲線。
    改變mu會水平移動曲線，而改變sigma會改變曲線的寬度和高度。
    """

if __name__ == "__main__":
    main()

print("123".__len__())