import torch

# tensor is just an array - multidimensional array
x = torch.Tensor([5,3])
y = torch.Tensor([2,1])
print(x*y)

# specifying shape
x = torch.zeros([2,5])
print(x)
print(x.shape)

# random value tensor generation
y = torch.rand([2,5])
print(y)

# Reshaping for flattening the multidimensional data so it could be fed to the Neural Network
y = y.view([1,10])
print(y)

