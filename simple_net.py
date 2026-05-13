import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 10),
            nn.Sigmoid(),
            nn.Linear(10, 100),
            nn.Sigmoid(),
            nn.Linear(100, 10),
            nn.Sigmoid(),
            nn.Linear(10, 1),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.loss = nn.MSELoss()
        self.net.to(device)
        self.predict = None

    def forward(self, x):
        return self.net(x)

    def model_train(self, train_loader, train_epochs):
        for epoch in range(train_epochs):
            for x, y in train_loader:
                y_hat = self.forward(x)
                loss = self.loss(y_hat, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if epoch % 100 == 0:
                print(f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}")


def main():

    original_data = np.linspace(-5 * np.pi, 5 * np.pi, 1000)
    original_data = original_data[:, None].astype(np.float32)
    noise = np.random.normal(0, 0.1, (1000, 1)).astype(np.float32)

    true_data = np.sin(original_data) + noise

    original_data_tensor = torch.from_numpy(original_data).to(device)
    true_data_tensor = torch.from_numpy(true_data).to(device)

    train_dataset = TensorDataset(original_data_tensor, true_data_tensor)
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

    model = SimpleNet()
    model.model_train(train_loader, 5000)

    with torch.no_grad():
        model.predict = model(original_data_tensor)

    plt.figure(figsize=(12, 7), dpi=160)
    plt.plot(
        original_data.reshape(-1),
        true_data.reshape(-1),
        label="True Data"
    )
    plt.plot(
        original_data,
        model.predict.detach().cpu().numpy(),
        label="Predicted Data",
    )
    plt.xlabel("x")
    plt.ylabel("sin(x) + noise")
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.legend(fontsize=15)
    plt.show()

    torch.save(model.state_dict(), "simple_sin.pt")

    clone = SimpleNet()
    clone.load_state_dict(torch.load("simple_sin.pt"))



if __name__ == "__main__":
    main()
