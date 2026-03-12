
import os
import json
import time
import timeit
import numpy as np

from cs336_basics.model import *
from cs336_basics.optimizer import *
from cs336_basics.nn_utils import *

import torch
# print(torch.__version__)

# torch.set_float32_matmul_precision('high')

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# scaled_dot_product_attention = annotated_scaled_dot_product_attention

################################

small = {'batch_size': 1, 'vocab_size': 50257, 'context_length': 64, 'd_model': 768,  'num_layers': 12, 'num_heads': 12, 'd_ff': 3072,  'rope_theta': 10000}
med   = {'batch_size': 1, 'vocab_size': 50257, 'context_length': 64, 'd_model': 1024, 'num_layers': 24, 'num_heads': 16, 'd_ff': 4096,  'rope_theta': 10000}
large = {'batch_size': 1, 'vocab_size': 50257, 'context_length': 64, 'd_model': 1280, 'num_layers': 36, 'num_heads': 20, 'd_ff': 5120,  'rope_theta': 10000}
xl    = {'batch_size': 1, 'vocab_size': 50257, 'context_length': 64, 'd_model': 1600, 'num_layers': 48, 'num_heads': 25, 'd_ff': 6400,  'rope_theta': 10000}
_27B  = {'batch_size': 1, 'vocab_size': 50257, 'context_length': 64, 'd_model': 2560, 'num_layers': 32, 'num_heads': 32, 'd_ff': 10240, 'rope_theta': 10000}

config = small

m = BasicsTransformerLM(
vocab_size=config['vocab_size'],
context_length=config['context_length'],
d_model=config['d_model'],
num_layers=config['num_layers'],
num_heads=config['num_heads'],
d_ff=config['d_ff'],
rope_theta=config['rope_theta'],
).to('cuda')

# simply calling compile can give 50% better performance.
m.compile()

optimizer = AdamW(m.parameters(), lr=1e-4)

################################

def fake_train(N):
  inf = []
  grad = []
  update = []

  for _ in range(N):
    x = torch.randint(low=0, high=config['vocab_size'], size=(config['batch_size'], config['context_length'])).to('cuda')
    y = torch.randint(low=0, high=config['vocab_size'], size=(config['batch_size'], config['context_length'])).to('cuda')
    
    T1 = time.time()
    p = m(x)
    T2 = time.time()
    loss = cross_entropy(p, y)
    loss = loss.backward()
    T3 = time.time()
    optimizer.step()
    T4 = time.time()

    inf.append( T2 - T1 )
    grad.append( T3 - T2 )
    update.append( T4 - T3 )

    torch.cuda.synchronize()

  return inf, grad, update

################################

# warmup
# inf, grad, update = fake_train(5)

# run 1 steps
inf, grad, update = fake_train(10)
print ('inf', np.mean(inf), np.std(inf))
print ('grad', np.mean(grad), np.std(grad))
print ('update', np.mean(update), np.std(update))

################################

'''
b)
inf 0.005195903778076172 3.44152573579595e-05
grad 0.005629181861877441 5.240874775765193e-05
update 0.015578889846801757 0.00013407352574959681

std is low.

c)
inf 0.22316055297851561 0.6535703970065881
grad 0.022184872627258302 0.0477221362080594
update 0.016055917739868163 0.004404501205468649
'''

################################

