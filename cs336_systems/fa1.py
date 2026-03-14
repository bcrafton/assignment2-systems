
import torch
from einops import rearrange, einsum
import math

class FlashAttention(torch.autograd.Function):
  def forward(ctx, Q, K, V, is_causal):
    # Your implementation should take input Q, K, and V as well as a flag is_causal and produce the output O and the logsumexp value L
    # is_causal --> ignore.
    # The autograd.Function forward should then use save L, Q, K, V, O for the backward pass and return O
    # So my interpretation is that we need to produce O, L ... but only return O. Save L somehow. Using context I guess?
    # AI: This ctx object is used to stash information (tensors, non-tensor data) in the forward pass that will be needed later in the backward pass using methods like ctx.save_for_backward()
    # AI: Causal self-attention (or masked self-attention) is a Transformer mechanism where each token in a sequence attends only to itself and previous tokens, blocking information from future positions
    # 
    # print ()
    # print (ctx)
    # pass
    # 
    # S = QK.T / d
    # Pij = softmaxj (S)ij
    # O = PV
    # 
    d_k = K.shape[-1]
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
    attention_weights = torch.softmax(attention_scores, dim=-1)
    O = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    L = torch.logsumexp(attention_scores, dim=-1)
    ctx.save_for_backward(L)
    return O

  def backward():
    raise NotImplementedError
