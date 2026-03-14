
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

small = {'name': 'small', 'vocab_size': 50257, 'd_model': 768,  'num_layers': 12, 'num_heads': 12, 'd_ff': 3072,  'rope_theta': 10000}
med   = {'name': 'med',   'vocab_size': 50257, 'd_model': 1024, 'num_layers': 24, 'num_heads': 16, 'd_ff': 4096,  'rope_theta': 10000}
large = {'name': 'large', 'vocab_size': 50257, 'd_model': 1280, 'num_layers': 36, 'num_heads': 20, 'd_ff': 5120,  'rope_theta': 10000}
xl    = {'name': 'xl',    'vocab_size': 50257, 'd_model': 1600, 'num_layers': 48, 'num_heads': 25, 'd_ff': 6400,  'rope_theta': 10000}
_27B  = {'name': '2.7B',  'vocab_size': 50257, 'd_model': 2560, 'num_layers': 32, 'num_heads': 32, 'd_ff': 10240, 'rope_theta': 10000}

small = {'name': 'small', 'vocab_size': 50257, 'd_model': 768,  'num_layers': 12, 'num_heads': 12, 'd_ff': 3072,  'rope_theta': 10000}

################################

def fake_train(N, vocab_size, batch_size, context_length):
  inf = []
  grad = []
  update = []

  for _ in range(N):
    x = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length)).to('cuda')
    y = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length)).to('cuda')
    
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

# for config in [small, med, large, xl, _27B]:
#   for context_length in [128, 256, 512, 1024]:

for config in [small]:
  for context_length in [128]:

    print (config['name'], context_length)

    try:
      m = BasicsTransformerLM(
      vocab_size=config['vocab_size'],
      context_length=context_length,
      d_model=config['d_model'],
      num_layers=config['num_layers'],
      num_heads=config['num_heads'],
      d_ff=config['d_ff'],
      rope_theta=config['rope_theta'],
      ).to('cuda')
      # m.compile()
      optimizer = AdamW(m.parameters(), lr=1e-4)

      # warmup
      inf, grad, update = fake_train(5, vocab_size=config['vocab_size'], batch_size=1, context_length=context_length)

      torch.cuda.memory._record_memory_history(max_entries=1000000)

      # run 1 steps
      inf, grad, update = fake_train(N=1, vocab_size=config['vocab_size'], batch_size=1, context_length=context_length)
      print ('inf', np.mean(inf), np.std(inf))
      print ('grad', np.mean(grad), np.std(grad))
      print ('update', np.mean(update), np.std(update))

      torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
      torch.cuda.memory._record_memory_history(enabled=None)

    except:
      print ('Ran out of memory')

################################

