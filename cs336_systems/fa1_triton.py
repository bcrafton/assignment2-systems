
import torch
from einops import rearrange, einsum
import math

import triton
import triton.language as tl

######################

def cdiv(x, y):
    return (x + y - 1) // y

######################

@triton.jit
def flash_fwd_kernel(
  Q_ptr, K_ptr, V_ptr,
  O_ptr, L_ptr,
  S_ptr,
  stride_qb, stride_qq, stride_qd,
  stride_kb, stride_kk, stride_kd,
  stride_vb, stride_vk, stride_vd,
  stride_ob, stride_oq, stride_od,
  stride_lb, stride_lq,
  stride_sb, stride_sq, stride_sk,
  N_QUERIES, N_KEYS,
  scale,
  D: tl.constexpr,
  Q_TILE_SIZE: tl.constexpr,
  K_TILE_SIZE: tl.constexpr,
  ):

  # Program indices
  batch_index = tl.program_id(0)
  query_tile_index = tl.program_id(1)

  # Offset each pointer with the corresponding batch index multiplied with the batch stride for each tensor
  Q_block_ptr = tl.make_block_ptr(
  Q_ptr + batch_index * stride_qb,
  shape=(N_QUERIES, D),
  strides=(stride_qq, stride_qd),
  offsets=(query_tile_index * Q_TILE_SIZE, 0),
  block_shape=(Q_TILE_SIZE, D),
  order=(1, 0),
  )

  # To debug, we suggest comparing the results of each Triton operation you perform with the tiled PyTorch implementation you wrote in part (a).
  # Your launch grid should be set as (Tq , batch_size), meaning each Triton program instance will load only elements from a single batch index, and only read/write to a single query tile of Q, O, and L.
  # The kernel should only have a single loop, which will iterate key tiles 1 ≤ j ≤ Tk.
  # Advance block pointers at the end of the loop.

  # "load only elements from a single batch index, and only read/write to a single query tile of Q, O, and L"
  # does that mean this kernel should load all the K tiles that it needs?
  # yes because look at the very next hint:
  # "The kernel should only have a single loop, which will iterate key tiles 1 ≤ j ≤ Tk"

  K_block_ptr = tl.make_block_ptr(
  K_ptr + batch_index * stride_kb,
  shape=(N_KEYS, D),
  strides=(stride_kk, stride_kd),
  offsets=(0 * K_TILE_SIZE, 0),
  block_shape=(K_TILE_SIZE, D),
  order=(1, 0),
  )

  V_block_ptr = tl.make_block_ptr(
  V_ptr + batch_index * stride_vb,
  shape=(N_KEYS, D),
  strides=(stride_vk, stride_vd),
  offsets=(0 * K_TILE_SIZE, 0),
  block_shape=(K_TILE_SIZE, D),
  order=(1, 0),
  )

  O_block_ptr = tl.make_block_ptr(
  O_ptr + batch_index * stride_ob,
  shape=(N_QUERIES, D),
  strides=(stride_oq, stride_od),
  offsets=(query_tile_index * Q_TILE_SIZE, 0),
  block_shape=(Q_TILE_SIZE, D),
  order=(1, 0),
  )

  L_block_ptr = tl.make_block_ptr(
  L_ptr + batch_index * stride_lb,
  shape=(N_QUERIES,),
  strides=(stride_lq,),
  offsets=(query_tile_index * Q_TILE_SIZE,),
  block_shape=(Q_TILE_SIZE,),
  order=(0,),
  )
  
  # print ('', stride_sb) # 16384
  # print ('', stride_sq) # 128
  # print ('', stride_sk) # 1

  # print ('', query_tile_index) # only [0,1,2,3]
  # print ('', batch_index) # [0,1,2,3,4,5,6,7] --> we must have these swapped.

  S_block_ptr = tl.make_block_ptr(
  S_ptr + batch_index * stride_sb,
  shape=(N_QUERIES, N_KEYS),
  strides=(stride_sq, stride_sk),
  offsets=(query_tile_index * Q_TILE_SIZE, 0),
  block_shape=(Q_TILE_SIZE, K_TILE_SIZE),
  order=(1, 0),
  )

  _Mij = tl.zeros(shape=(Q_TILE_SIZE,), dtype=tl.float32)
  Mij = tl.zeros(shape=(Q_TILE_SIZE,), dtype=tl.float32)
  Lij = tl.zeros(shape=(Q_TILE_SIZE,), dtype=tl.float32)
  Oij = tl.zeros(shape=(Q_TILE_SIZE,D), dtype=tl.float32)

  for k in range(0, N_KEYS // K_TILE_SIZE):
    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

    Sij = tl.dot(Qi, tl.trans(Kj)) * scale
    tl.store(S_block_ptr, Sij, boundary_check=(0,))

    Mij = tl.maximum(_Mij, tl.max(Sij, axis=-1))
    Pij = tl.exp(Sij - Mij.reshape(Q_TILE_SIZE, 1))
    Lij = tl.exp(_Mij - Mij) * Lij + tl.sum(Pij, axis=-1)

    K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
    V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    S_block_ptr = S_block_ptr.advance((0, K_TILE_SIZE))

    diag = tl.exp(_Mij - Mij).reshape(Q_TILE_SIZE, 1)
    Oij = Oij * diag + tl.dot(Pij, Vj)

    _Mij = Mij

  L = Mij + tl.log(Lij)
  tl.store(L_block_ptr, L, boundary_check=(0,))

  diag = 1. / Lij.reshape(Q_TILE_SIZE, 1)
  Oij = Oij * diag
  tl.store(O_block_ptr, Oij, boundary_check=(0,))

class FlashAttentionTriton(torch.autograd.Function):

  '''
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
  '''

  @staticmethod
  def forward(ctx, Q, K, V, is_causal):
    b_q, seq_q, d_q = Q.shape
    b_k, seq_k, d_k = K.shape
    b_v, seq_v, d_v = V.shape

    # print (b_q, seq_q, d_q)

    # Reshape input tensor to 2D
    # input_shape = x.shape
    # x = rearrange(x, "... d -> (...) d")

    # assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "Expected CUDA tensors"
    assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

    # ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16
    # ctx.ROWS_TILE_SIZE = 16 # Each thread processes 16 batch elements at a time
    # ctx.input_shape = input_shape

    # Need to initialize empty result tensor. Note that these elements are not necessarily 0!
    O = torch.zeros((b_v, seq_v, d_v), device='cuda')
    L = torch.zeros((b_q, seq_q), device='cuda')
    S = torch.zeros((b_q, seq_q, seq_k), device='cuda')

    # Launch our kernel with n instances in our 1D grid.
    # n_rows = y.numel()
    scale = 1 / math.sqrt(d_k)
    Q_TILE_SIZE = 16
    K_TILE_SIZE = 16
    # weighted_sum_backward[(cdiv(n_rows, ROWS_TILE_SIZE),)](
    # print (seq_k, K_TILE_SIZE)
    flash_fwd_kernel[(b_q, seq_k // K_TILE_SIZE)](
      Q, K, V,
      O, L,
      S,
      Q.stride(0), Q.stride(1), Q.stride(2),
      K.stride(0), K.stride(1), K.stride(2),
      V.stride(0), V.stride(1), V.stride(2),
      O.stride(0), O.stride(1), O.stride(2),
      L.stride(0), L.stride(1),
      S.stride(0), S.stride(1), S.stride(2),
      seq_q, seq_k,
      scale,
      d_k,
      Q_TILE_SIZE,
      K_TILE_SIZE
    )

    ctx.save_for_backward(L)
    return O

  def backward():
    raise NotImplementedError





