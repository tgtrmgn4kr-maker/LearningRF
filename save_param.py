import torch

x = torch.arange(4)

torch.save(x, 'x.pt')

y = torch.load('x.pt')

print(y)

a = torch.arange(4)

torch.save([x, a], 'a.pt')

c, d = torch.load('a.pt')

print(c)
print(d)