"""
v1_baseline: Multi-Head Attention Triton Kernel (No Optimization)
직접적인 1:1 변환, 최적화 없음
"""
import torch
import triton
import triton.language as tl
import math


@triton.jit
def mha_kernel_baseline(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, M, N, K,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Baseline multi-head attention kernel.
    Computes: O = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Args:
        Q_ptr: Query tensor [Z, H, M, K]
        K_ptr: Key tensor [Z, H, N, K]
        V_ptr: Value tensor [Z, H, N, K]
        O_ptr: Output tensor [Z, H, M, K]
        stride_*: Strides for each dimension
        Z: Batch size
        H: Number of heads
        M: Query sequence length
        N: Key/Value sequence length
        K: Head dimension
        scale: Scaling factor (1/sqrt(d_k))
        BLOCK_*: Block sizes
    """
    # Program indices
    pid_z = tl.program_id(0)  # Batch
    pid_h = tl.program_id(1)  # Head
    pid_m = tl.program_id(2)  # Query block
    
    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize pointers
    q_ptrs = Q_ptr + pid_z * stride_qz + pid_h * stride_qh + \
             offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    k_ptrs = K_ptr + pid_z * stride_kz + pid_h * stride_kh + \
             offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
    v_ptrs = V_ptr + pid_z * stride_vz + pid_h * stride_vh + \
             offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
    
    # Load Q block
    mask_m = offs_m < M
    mask_k = offs_k < K
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
    
    # Initialize accumulators
    m_i = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    
    # Loop over K/V blocks
    for block_n in range(0, N, BLOCK_N):
        offs_n_curr = block_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n_curr < N
        
        # Load K block
        k_ptrs_curr = K_ptr + pid_z * stride_kz + pid_h * stride_kh + \
                      offs_n_curr[:, None] * stride_kn + offs_k[None, :] * stride_kk
        k = tl.load(k_ptrs_curr, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        # Compute QK^T
        qk = tl.dot(q, tl.trans(k)) * scale  # [BLOCK_M, BLOCK_N]
        
        # Apply mask for out-of-bounds
        qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, -float('inf'))
        
        # Compute new max
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        # Compute attention weights
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        
        # Update accumulator
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        # Load V block
        v_ptrs_curr = V_ptr + pid_z * stride_vz + pid_h * stride_vh + \
                      offs_n_curr[:, None] * stride_vn + offs_k[None, :] * stride_vk
        v = tl.load(v_ptrs_curr, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        # Accumulate weighted V
        acc += tl.dot(p.to(v.dtype), v)
        
        m_i = m_new
    
    # Normalize
    acc = acc / l_i[:, None]
    
    # Store output
    o_ptrs = O_ptr + pid_z * stride_oz + pid_h * stride_oh + \
             offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(o_ptrs, acc, mask=mask_m[:, None] & mask_k[None, :])


def triton_mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Triton multi-head attention implementation (baseline).
    Computes: softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        v: Value tensor [batch, heads, seq_len, head_dim]
    
    Returns:
        Output tensor [batch, heads, seq_len, head_dim]
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.shape == k.shape == v.shape
    
    Z, H, M, K = q.shape
    N = k.shape[2]
    
    # Ensure contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    # Allocate output
    o = torch.empty_like(q)
    
    # Scaling factor
    scale = 1.0 / math.sqrt(K)
    
    # Block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = min(64, K)
    
    # Grid
    grid = (Z, H, triton.cdiv(M, BLOCK_M))
    
    mha_kernel_baseline[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Z, H, M, N, K,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return o
