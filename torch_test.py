import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

a = torch.arange(16).reshape(4, 4)
b = a.clone()
b = b.T
a.to(device)
b.to(device)

print(a)
print(b)

print("a * b:", a * b)  # element-wise multiplication
print("a @ b:", a @ b)  # matrix multiplication
print("a.sum():", a.sum())  # sum of all elements
print("a.sum(dim=0):", a.sum(dim=0))  # sum along columns
print("a.sum(dim=1):", a.sum(dim=1))  # sum along rows
print("a.prod():", a.prod())
print("a.mean(dtype=torch.float64):", a.mean(dtype=torch.float64))  # mean of all elements

M1 = torch.tensor([[1.,2.,3.,4.]])
M2 = torch.tensor([[5.],[6.],[7.],[8.]])
print("M1 @ M2:", M1 @ M2)  # matrix multiplication
print("M2 @ M1:", M2 @ M1)  # matrix multiplication
print("torch.norm(M1):", torch.norm(M1))  # Frobenius norm of M1
print("torch.norm(M2):", torch.norm(M2))  # Frobenius norm of M2

print(torch.seed()) # a 64-bit unsigned integer random number

print(torch.cat((a, b), dim=0)) # 豎排
print(torch.cat((a, b), dim=1)) # 橫排

