import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(10, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))


net = MLP()

# Save a model
x = torch.arange(10, dtype=torch.float32)

y = net(x)

torch.save(net.state_dict(), "model.params")

# Load a model
clone = MLP()
clone.load_state_dict(torch.load("model.params"))

y_clone = clone(x)

print(y==y_clone)