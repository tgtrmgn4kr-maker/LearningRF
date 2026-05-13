import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.optim = torch.optim.Adam(self.net.parameters(), lr=0.0001)
        self.loss = nn.MSELoss()
        self.predict = None
        self.net.to(device)

    def forward(self, x):
        return self.net(x)

    def model_train(self, train_epochs, train_loader):
        for epoch in range(train_epochs):
            for x, y in train_loader:
                y_hat = self.forward(x)
                loss = self.loss(y_hat, y)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            if epoch % 100 == 0:
                print(f"Epoch {epoch+1}: Loss {loss.item():.4f}")

        print(self.net[4].weight,"\n",self.net[4].bias)


def main():

    x = np.linspace(-2*np.pi, 2*np.pi, 1000)
    y = np.cos(x)
    z = np.sin(x)

    # 三者等價
    X = np.expand_dims(x, axis=1).astype(np.float32)
    Y = y.reshape(-1, 1).astype(np.float32)
    Z = z[:, None]
    print(x.shape, X.shape, y.shape, Y.shape, z.shape, Z.shape)
    X, Y = torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device)

    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, shuffle=True, batch_size=1000)

    net = Net()
    net.model_train(1000, loader)

    with torch.no_grad():
        net.predict = net(X) # (1000, 1)

    plt.figure(figsize=(12, 7), dpi=160)
    plt.plot(x, y, label='True Value', marker="X")
    if net.predict is not None:
        plt.plot(
            x[:,None],
            net.predict.detach().cpu().numpy(),
            label='Predict Value',
            marker="o"
        )
    plt.xlabel("x")
    plt.ylabel("cos(x)")
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.legend(fontsize=15)
    plt.show()


if __name__ == '__main__':
    main()