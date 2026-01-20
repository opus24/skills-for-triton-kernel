"""
v4_optimized: Multi-Head Attention Triton Kernel (Tiling + Memory Coalescing)
두 최적화 기법을 모두 적용한 Flash Attention 스타일 구현
"""
import torch
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N'],
)
@triton.jit
def mha_kernel_optimized(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, M, N, K,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr = 64,
):
    """
    Fully optimized Flash Attention kernel.
    
    Optimizations:
    1. Tiling: Process Q in BLOCK_M tiles, K/V in BLOCK_N tiles
    2. Memory Coalescing: Contiguous access patterns
    3. Online Softmax: O(1) memory for attention weights
    4. Auto-tuning: Dynamic block size selection
    5. Pipeline stages: Overlapped memory access
    """
    pid_z = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    # Coalesced offset computation
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Base pointers
    q_base = Q_ptr + pid_z * stride_qz + pid_h * stride_qh
    k_base = K_ptr + pid_z * stride_kz + pid_h * stride_kh
    v_base = V_ptr + pid_z * stride_vz + pid_h * stride_vh
    
    # Load Q tile (coalesced)
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    mask_m = offs_m < M
    mask_k = offs_k < K
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
    
    # Online softmax accumulators
    m_i = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    
    # Tiled + coalesced K/V processing
    num_tiles = tl.cdiv(N, BLOCK_N)
    for tile_idx in range(num_tiles):
        offs_n = tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        
        # Coalesced K load
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        # Scaled dot product
        qk = tl.dot(q, tl.trans(k)) * scale
        qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, -float('inf'))
        
        # Online softmax update (numerically stable)
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        # Correction factor for previous accumulator
        alpha = tl.exp(m_i - m_new)
        
        # Current attention weights
        p = tl.exp(qk - m_new[:, None])
        
        # Update running sum and accumulator
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        # Coalesced V load
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        # Weighted value accumulation
        acc += tl.dot(p.to(v.dtype), v)
        
        m_i = m_new
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Coalesced output store
    o_base = O_ptr + pid_z * stride_oz + pid_h * stride_oh
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(o_ptrs, acc, mask=mask_m[:, None] & mask_k[None, :])


@triton.jit
def mha_kernel_optimized_causal(
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
    Optimized causal attention kernel.
    Applies causal mask (upper triangular) efficiently.
    """
    pid_z = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    
    q_base = Q_ptr + pid_z * stride_qz + pid_h * stride_qh
    k_base = K_ptr + pid_z * stride_kz + pid_h * stride_kh
    v_base = V_ptr + pid_z * stride_vz + pid_h * stride_vh
    
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    mask_m = offs_m < M
    mask_k = offs_k < K
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
    
    m_i = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    
    # Only process up to the diagonal for causal attention
    max_tile = tl.cdiv((pid_m + 1) * BLOCK_M, BLOCK_N)
    for tile_idx in range(max_tile):
        offs_n = tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        qk = tl.dot(q, tl.trans(k)) * scale
        
        # Apply causal mask
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(mask_m[:, None] & mask_n[None, :] & causal_mask, qk, -float('inf'))
        
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new
    
    acc = acc / l_i[:, None]
    
    o_base = O_ptr + pid_z * stride_oz + pid_h * stride_oh
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(o_ptrs, acc, mask=mask_m[:, None] & mask_k[None, :])


def triton_mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """
    Triton multi-head attention with full optimization.
    Flash Attention style implementation with tiling and coalescing.
    
    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        v: Value tensor [batch, heads, seq_len, head_dim]
        causal: Whether to apply causal mask (default: False)
    
    Returns:
        Output tensor [batch, heads, seq_len, head_dim]
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.shape == k.shape == v.shape
    
    Z, H, M, K = q.shape
    N = k.shape[2]
    
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    o = torch.empty_like(q)
    scale = 1.0 / math.sqrt(K)
    
    # Determine BLOCK_K based on head dimension
    BLOCK_K = triton.next_power_of_2(K)
    BLOCK_K = max(BLOCK_K, 16)
    BLOCK_K = min(BLOCK_K, 128)
    
    grid = lambda meta: (Z, H, triton.cdiv(M, meta['BLOCK_M']))
    
    if causal:
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = min(64, K)
        grid_fixed = (Z, H, triton.cdiv(M, BLOCK_M))
        mha_kernel_optimized_causal[grid_fixed](
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
    else:
        mha_kernel_optimized[grid](
            q, k, v, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            Z, H, M, N, K,
            scale,
            BLOCK_K=BLOCK_K,
        )
    
    return o
