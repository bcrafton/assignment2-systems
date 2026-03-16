
import torch

x = torch.Tensor([1.0, 2.0, 3.0, 4.0])
y = torch.logsumexp(x, dim=-1)
# print (y)

softmax = torch.softmax(x, dim=-1)
print (softmax)

softmax = torch.exp(x) / torch.sum(torch.exp(x))
print (softmax)

softmax = torch.exp(x - torch.max(x)) / torch.sum(torch.exp(x - torch.max(x)))
print (softmax)
