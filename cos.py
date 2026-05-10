import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
import numpy as np
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
        self.optim = torch.optim.Adam(self.parameters(), lr=0.01)
        self.loss = nn.MSELoss()


    def forward(self, x):
        return self.net(x)
    def train(self, train_epochs, train_loader):
        for epoch in range(train_epochs):
            for x, y in train_loader:
                y_hat = self.forward(x)
                loss = self.loss(y_hat, y)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            print(f"Epoch {epoch+1}: Loss {loss.item():.4f}")






x = np.linspace(-2*np.pi, 2*np.pi, 1000)
y = np.cos(x)
z = np.sin(x)

# 三者等價
X = np.expand_dims(x, axis=1)
Y = y.reshape(-1, 1)
Z = z[:, None]
print(x.shape, X.shape, y.shape, Y.shape, z.shape, Z.shape)

dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
loader = DataLoader(dataset, batch_size=100, shuffle=True)



