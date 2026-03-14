
import os
import json
import time
import timeit
import numpy as np

from cs336_basics.model import *
from cs336_basics.optimizer import *
from cs336_basics.nn_utils import *

import torch

################################

def fake_train(N, batch_size, context_length, d_model):
  inf = []
  grad = []
  update = []

  for _ in range(N):
    x = torch.normal(mean=0, std=1, size=(batch_size, context_length, d_model)).to('cuda')
    y = torch.normal(mean=0, std=1, size=(batch_size, context_length, d_model)).to('cuda')
    
    T1 = time.time()
    p = m(x, None)
    
    T2 = time.time()
    loss = torch.mean(p - y)
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

for d_model in [16, 32, 64, 128]:
  for context_length in [256, 1024, 4096, 8192, 16384]:

    print (d_model, context_length)

    try:
      positional_encoder = RotaryEmbedding(context_length=context_length, dim=d_model, theta=10000)
      m = CausalMultiHeadSelfAttention(d_model=d_model, num_heads=1, positional_encoder=positional_encoder).to('cuda')
      m.compile()
      optimizer = AdamW(m.parameters(), lr=1e-4)

      # warmup
      inf, grad, update = fake_train(N=5, batch_size=8, context_length=context_length, d_model=d_model)

      torch.cuda.memory._record_memory_history(max_entries=1000000)

      # run 1 steps
      inf, grad, update = fake_train(N=100, batch_size=8, context_length=context_length, d_model=d_model)
      print ('inf', np.mean(inf), np.std(inf))
      print ('grad', np.mean(grad), np.std(grad))
      print ('update', np.mean(update), np.std(update))

      torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
      torch.cuda.memory._record_memory_history(enabled=None)
    except:
      print ('Ran out of memory')

################################

