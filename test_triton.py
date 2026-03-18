
import torch
import numpy as np

from cs336_systems.fa1_triton import FlashAttentionTriton
from cs336_systems.fa1_hack import FlashAttention

#################################

Q = torch.ones(size=(4, 128, 64)).to('cuda')
K = torch.ones(size=(4, 128, 64)).to('cuda')
V = torch.ones(size=(4, 128, 64)).to('cuda')

Q = torch.normal(mean=0., std=1., size=(4, 128, 64)).to('cuda')
K = torch.normal(mean=0., std=1., size=(4, 128, 64)).to('cuda')
V = torch.normal(mean=0., std=1., size=(4, 128, 64)).to('cuda')

is_causal = False

#################################

O_ref = FlashAttention.apply(Q, K, V, is_causal)
O = FlashAttentionTriton.apply(Q, K, V, is_causal)

#################################
'''
for i in range(0, 128, 16):
  # print (O[1, i:i+16, 0:16])
  pass

for i in range(0, 64, 16):
  #print (O[1, 0:16, i:i+16])
  pass
'''
#################################

print (O_ref.shape)
print (O_ref[0][0][0:4])

print (O.shape)
print (O[0][0][0:4])

#################################

O_ref = O_ref.cpu().numpy()
O = O.cpu().numpy()
assert np.all(np.isclose(O, O_ref, atol=1e-2))

#################################
