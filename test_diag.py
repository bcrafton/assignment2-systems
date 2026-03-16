import torch

x = torch.Tensor([
[1.0, 2.0],
[3.0, 4.0],
])

print (x)
print (x.shape)
print ()

y = torch.diag_embed(input=x, offset=0, dim1=-2, dim2=-1)

print (y)
print (y.shape)
print ()
