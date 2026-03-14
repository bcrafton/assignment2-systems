
import os
import json
import time
import timeit
import numpy as np

from cs336_basics.model import *
from cs336_basics.optimizer import *
from cs336_basics.nn_utils import *

import torch

####################################

class ToyModel(nn.Module):

  def __init__(self, in_features: int, out_features: int):
    super().__init__()
    self.fc1 = nn.Linear(in_features, 10, bias=False)
    self.ln = nn.LayerNorm(10)
    self.fc2 = nn.Linear(10, out_features, bias=False)
    self.relu = nn.ReLU()

  def forward(self, x):
    print ('fc1.w', self.fc1.weight.dtype)
    x = self.fc1(x); print ('fc1', x.dtype)
    x = self.relu(x); print ('relu', x.dtype)
    x = self.ln(x); print ('ln', x.dtype)
    print ('fc2.w', self.fc2.weight.dtype)
    x = self.fc2(x); print ('fc2', x.dtype)
    return x

####################################

x = torch.Tensor(np.random.normal(loc=0, scale=1, size=100)).to('cuda')
y = torch.Tensor(np.random.normal(loc=0, scale=1, size=10)).to('cuda')

m = ToyModel(100, 10).to('cuda')
optimizer = AdamW(m.parameters(), lr=1e-4)

# with torch.autocast(device_type="cuda", dtype=torch.float16):
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
  _y = m(x)
  loss = torch.mean(_y - y)
  print ('loss', loss.dtype)
  loss = loss.backward()
  print ('fc1 grad', m.fc1.weight.grad.dtype)
  optimizer.step()

####################################
