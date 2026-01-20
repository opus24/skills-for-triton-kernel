"""
v2_tiling: Multi-Head Attention Triton Kernel with Tiling Optimization
Tiling 기법을 적용하여 캐시 효율성 향상 (Flash Attention 스타일)
"""
import torch
import triton
import triton.language as tl
import math


@triton.jit
def mha_kernel_tiling(
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
    Tiled multi-head attention kernel (Flash Attention style).
    Uses online softmax computation for memory efficiency.
    
    Tiling strategy:
    - Process Q in BLOCK_M chunks
    - Process K/V in BLOCK_N chunks
    - Accumulate attention incrementally
    """
    pid_z = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize Q pointers
    q_ptrs = Q_ptr + pid_z * stride_qz + pid_h * stride_qh + \
             offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    
    mask_m = offs_m < M
    mask_k = offs_k < K
    
    # Load Q tile
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
    
    # Online softmax accumulators
    m_i = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    
    # Tiled iteration over K/V
    num_tiles = tl.cdiv(N, BLOCK_N)
    for tile_idx in range(num_tiles):
        offs_n = tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        
        # Load K tile
        k_ptrs = K_ptr + pid_z * stride_kz + pid_h * stride_kh + \
                 offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        # Compute attention scores for this tile
        qk = tl.dot(q, tl.trans(k)) * scale
        qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, -float('inf'))
        
        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        # Load V tile
        v_ptrs = V_ptr + pid_z * stride_vz + pid_h * stride_vh + \
                 offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        # Accumulate weighted values
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store output tile
    o_ptrs = O_ptr + pid_z * stride_oz + pid_h * stride_oh + \
             offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(o_ptrs, acc, mask=mask_m[:, None] & mask_k[None, :])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def mha_kernel_tiling_autotune(
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
    """Auto-tuned tiled MHA kernel."""
    pid_z = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    
    q_ptrs = Q_ptr + pid_z * stride_qz + pid_h * stride_qh + \
             offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    
    mask_m = offs_m < M
    mask_k = offs_k < K
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
    
    m_i = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    
    num_tiles = tl.cdiv(N, BLOCK_N)
    for tile_idx in range(num_tiles):
        offs_n = tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        
        k_ptrs = K_ptr + pid_z * stride_kz + pid_h * stride_kh + \
                 offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        qk = tl.dot(q, tl.trans(k)) * scale
        qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, -float('inf'))
        
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        v_ptrs = V_ptr + pid_z * stride_vz + pid_h * stride_vh + \
                 offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new
    
    acc = acc / l_i[:, None]
    
    o_ptrs = O_ptr + pid_z * stride_oz + pid_h * stride_oh + \
             offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(o_ptrs, acc, mask=mask_m[:, None] & mask_k[None, :])


def triton_mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Triton multi-head attention with tiling optimization.
    Uses Flash Attention style online softmax for memory efficiency.
    
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
    
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    o = torch.empty_like(q)
    scale = 1.0 / math.sqrt(K)
    
    # Determine BLOCK_K based on head dimension
    BLOCK_K = triton.next_power_of_2(K)
    BLOCK_K = max(BLOCK_K, 16)
    BLOCK_K = min(BLOCK_K, 128)
    
    # Use autotuned kernel
    grid = lambda meta: (Z, H, triton.cdiv(M, meta['BLOCK_M']))
    
    mha_kernel_tiling_autotune[grid](
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
