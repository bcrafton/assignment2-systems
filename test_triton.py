
import torch

from cs336_systems.fa1_triton import FlashAttentionTriton
from cs336_systems.fa1 import FlashAttention

#################################

Q = torch.ones(size=(4, 128, 64)).to('cuda')
K = torch.ones(size=(4, 128, 64)).to('cuda')
V = torch.ones(size=(4, 128, 64)).to('cuda')

# Q = torch.normal(mean=0., std=1., size=(4, 128, 64)).to('cuda')
# K = torch.normal(mean=0., std=1., size=(4, 128, 64)).to('cuda')
# V = torch.normal(mean=0., std=1., size=(4, 128, 64)).to('cuda')

is_causal = False

O = FlashAttention.apply(Q, K, V, is_causal)
# print (O)
print (O.shape)
print (O[0][0][0])

O = FlashAttentionTriton.apply(Q, K, V, is_causal)
# print (O)
print (O.shape)
print (O[0][0][0])

# print (O[1, :, :])

# print (O[0,  0:16,  0:16])
# print (O[0,  0:16, 16:32])

# print (O[0,  0:16,  0:16])
# print (O[0, 16:32,  0:16])
# print (O[0, 32:48,  0:16])
# print (O[0, 48:64,  0:16])
# print (O[0, 64:80,  0:16])

for i in range(0, 128, 16):
  # print (O[1, i:i+16, 0:16])
  pass

for i in range(0, 64, 16):
  # print (O[1, 0:16, i:i+16])
  pass

#################################
