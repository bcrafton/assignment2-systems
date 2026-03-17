
import torch
from einops import rearrange, einsum
import math

class FlashAttentionTriton(torch.autograd.Function):

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
    S = torch.zeros((b_q, seq_q, seq_k), device='cuda')
    L = torch.zeros((b_q, seq_q), device='cuda')
    O = torch.zeros((b_v, seq_v, d_v), device='cuda')

    # (2) Compute attention over blocks
    for i in range(num_block_q):
      _Mij = torch.zeros((b_q, block_q), device='cuda')
      _Lij = torch.zeros((b_q, block_q), device='cuda')
      _Oij = torch.zeros((b_q, block_q, d_q), device='cuda')
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
    return O

  def backward():
    raise NotImplementedError





