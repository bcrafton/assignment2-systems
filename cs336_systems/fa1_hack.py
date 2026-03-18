
import torch
from einops import rearrange, einsum
import math

'''
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
'''

class FlashAttention(torch.autograd.Function):

  def forward(ctx, Q, K, V, is_causal):
    # Set block sizes
    block_q = 16
    block_k = 16

    # Get dimensions of tensors
    b_q, seq_q, d_q = Q.shape
    b_k, seq_k, d_k = K.shape
    b_v, seq_v, d_v = V.shape

    num_block_q = seq_q // block_q
    num_block_k = seq_k // block_k

    # (1) Initialize outputs O & L
    S = torch.zeros((b_q, seq_q, seq_k)).to('cuda')
    L = torch.zeros((b_q, seq_q)).to('cuda')
    O = torch.zeros((b_v, seq_v, d_v)).to('cuda')

    # (2) Compute attention over blocks
    for i in range(num_block_q):
      _Mij = torch.zeros((b_q, block_q)).to('cuda')
      _Lij = torch.zeros((b_q, block_q)).to('cuda')
      _Oij = torch.zeros((b_q, block_q, d_q)).to('cuda')
      for j in range(num_block_k):
        Qi = Q[:, i*block_q : (i+1)*block_q, :]
        Kj = K[:, j*block_k : (j+1)*block_k, :]
        Vj = V[:, j*block_k : (j+1)*block_k, :]
        Sij = einsum(Qi, Kj, "... seq_q d, ... seq_k d -> ... seq_q seq_k") / math.sqrt(d_k)
        S[:, i*block_q : (i+1)*block_q, j*block_k : (j+1)*block_k] = Sij

        Mij = torch.maximum( _Mij, torch.amax(Sij, axis=-1) )
        Pij = torch.exp(Sij - Mij.unsqueeze(-1))
        # the paper does say this operation is "pointwise":
        Lij = torch.exp(_Mij - Mij) * _Lij + torch.sum(Pij, axis=-1)

        diag = torch.diag_embed(input=torch.exp(_Mij - Mij), offset=0, dim1=-2, dim2=-1)
        Oij = einsum(diag, _Oij, "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v") + einsum(Pij, Vj, "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v")

        _Mij = Mij
        _Lij = Lij
        _Oij = Oij

      L[:, i*block_q : (i+1)*block_q] = Mij + torch.log(Lij)

      diag = torch.diag_embed(input=Lij**-1, offset=0, dim1=-2, dim2=-1)
      Oij = einsum(diag, Oij, "... seq_q seq_q, ... seq_q d_q -> ... seq_q d_q")
      O[:, i*block_q : (i+1)*block_q, :] = Oij

    ctx.save_for_backward(L)
    return L

  def backward():
    raise NotImplementedError

'''
class FlashAttention(torch.autograd.Function):

  def forward(ctx, Q, K, V, is_causal):
    # Set block sizes
    block_q = 16
    block_k = 16

    # Get dimensions of tensors
    b_k, seq_k, d_k = K.shape
    b_q, seq_q, d_q = Q.shape
    b_v, seq_v, d_v = V.shape

    num_block_q = seq_q // block_q
    num_block_k = seq_k // block_k

    # (1) Initialize outputs O & L
    O = torch.zeros_like(V)
    L = torch.zeros((b_q, seq_q))

    # (2) Compute attention over blocks
    for i in range(num_block_q):
      _Oij = torch.zeros((b_q, block_q, d_q))
      _Lij = torch.zeros((b_q, block_q))
      _Mij = torch.zeros((b_q, block_q))
      for j in range(num_block_k):
        Qi = Q[:, i*block_q : (i+1)*block_q, :]
        Kj = K[:, j*block_k : (j+1)*block_k, :]
        Vj = V[:, j*block_k : (j+1)*block_k, :]
        Sij = einsum(Qi, Kj, "... seq_q d_q, ... seq_k d_k -> ... seq_q seq_k") / math.sqrt(d_k)
        Mij = torch.maximum( _Mij, torch.amax(Sij, axis=-1) )
        Pij = torch.exp(Sij - Mij.unsqueeze(-1))
        # the paper does say this operation is "pointwise":
        Lij = torch.exp(_Mij - Mij) * _Lij + torch.sum(Pij, axis=-1)
        diag = torch.diag_embed(input=torch.exp(_Mij - Mij), offset=0, dim1=-2, dim2=-1)
        Oij = einsum(diag, _Oij, "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v") + einsum(Pij, Vj, "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v")
        _Oij = Oij
        _Mij = Mij

      tmp = Lij.unsqueeze(-1) * torch.eye(block_q, block_q)
      Oij = einsum(tmp, Oij, "... seq_q seq_q, ... seq_q d_q -> ... seq_q d_q")
      O[:, i*block_q : (i+1)*block_q, :] = Oij
      L[:, i*block_q : (i+1)*block_q] = Lij

    ctx.save_for_backward(L)
    return O

  def backward():
    raise NotImplementedError
'''




