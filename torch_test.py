import torch

a = torch.arange(16).reshape(4, 4)
print(a)

b = a.clone()
b = b.T

print(b)

print(a * b)  # element-wise multiplication
print(a @ b)  # matrix multiplication
print(a.sum())  # sum of all elements
print(a.sum(dim=0))  # sum along columns
print(a.sum(dim=1))  # sum along rows
print(a.mean(dtype=torch.float64))  # mean of all elements

M1 = torch.tensor([[1.,2.,3.,4.]])
M2 = torch.tensor([[5.],[6.],[7.],[8.]])
print(M1 @ M2)  # matrix multiplication
print(M2 @ M1)  # matrix multiplication
print(torch.norm(M1))  # Frobenius norm of M1
print(torch.norm(M2))  # Frobenius norm of M2
