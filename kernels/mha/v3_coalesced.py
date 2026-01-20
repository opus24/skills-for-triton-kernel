"""
v3_coalesced: Multi-Head Attention Triton Kernel with Memory Coalescing
Memory Coalescing 기법을 적용하여 메모리 대역폭 활용률 향상
"""
import torch
import triton
import triton.language as tl
import math


@triton.jit
def mha_kernel_coalesced(
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
    Memory coalesced multi-head attention kernel.
    Optimizes memory access patterns for better bandwidth utilization.
    
    Memory coalescing strategy:
    - Contiguous thread access to Q, K, V
    - Aligned memory loads/stores
    - Vectorized operations where possible
    """
    pid_z = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    # Coalesced offset computation
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Base pointers for coalesced access
    q_base = Q_ptr + pid_z * stride_qz + pid_h * stride_qh
    k_base = K_ptr + pid_z * stride_kz + pid_h * stride_kh
    v_base = V_ptr + pid_z * stride_vz + pid_h * stride_vh
    
    # Coalesced Q load
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    mask_m = offs_m < M
    mask_k = offs_k < K
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
    
    # Online softmax state
    m_i = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    
    # Process K/V with coalesced access
    for block_n in range(0, N, BLOCK_N):
        offs_n = block_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        
        # Coalesced K load
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        # Compute QK^T
        qk = tl.dot(q, tl.trans(k)) * scale
        qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, -float('inf'))
        
        # Online softmax
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        # Coalesced V load
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new
    
    # Normalize
    acc = acc / l_i[:, None]
    
    # Coalesced output store
    o_base = O_ptr + pid_z * stride_oz + pid_h * stride_oh
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(o_ptrs, acc, mask=mask_m[:, None] & mask_k[None, :])


def triton_mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Triton multi-head attention with memory coalescing.
    Ensures contiguous memory access for optimal bandwidth.
    
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
    
    # Ensure contiguous memory layout
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    o = torch.empty_like(q)
    scale = 1.0 / math.sqrt(K)
    
    # Block sizes optimized for coalescing
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = min(64, K)
    
    grid = (Z, H, triton.cdiv(M, BLOCK_M))
    
    mha_kernel_coalesced[grid](
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
