"""
v4_optimized: Add Triton Kernel (Tiling + Memory Coalescing)
두 기법을 조합하여 최대 성능 추구
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=4),
    ],
    key=['n_elements'],
)
@triton.jit
def add_kernel_optimized(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fully optimized add kernel combining tiling and memory coalescing.
    Computes output = x + alpha * y
    
    Optimization strategies:
    1. Tiling: Process data in optimal BLOCK_SIZE chunks
    2. Memory Coalescing: Consecutive thread access patterns
    3. Auto-tuning: Dynamic BLOCK_SIZE selection
    4. Pipeline stages: Overlapped memory access and computation
    
    Args:
        x_ptr: First input tensor pointer (contiguous)
        y_ptr: Second input tensor pointer (contiguous)
        output_ptr: Output tensor pointer (contiguous)
        n_elements: Total number of elements
        alpha: Scalar multiplier for y
        BLOCK_SIZE: Optimal tile size (auto-tuned)
    """
    pid = tl.program_id(0)
    
    # Tiled + coalesced memory access
    tile_start = pid * BLOCK_SIZE
    offsets = tile_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Coalesced vector loads
    x_data = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_data = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Fused multiply-add operation
    result = x_data + alpha * y_data
    
    # Coalesced vector store
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def add_kernel_optimized_fixed(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized add kernel with fixed BLOCK_SIZE (no autotune overhead).
    For use when optimal BLOCK_SIZE is known.
    """
    pid = tl.program_id(0)
    
    tile_start = pid * BLOCK_SIZE
    offsets = tile_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x_data = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_data = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    result = x_data + alpha * y_data
    
    tl.store(output_ptr + offsets, result, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Triton add implementation with full optimization.
    Combines tiling and memory coalescing for maximum performance.
    Computes: x + alpha * y
    
    Optimizations applied:
    - Auto-tuned BLOCK_SIZE selection
    - Memory coalescing for bandwidth optimization
    - Pipeline stages for latency hiding
    - Contiguous memory access patterns
    
    Args:
        x: First input tensor
        y: Second input tensor
        alpha: Scalar multiplier for y (default: 1.0)
    
    Returns:
        Result tensor: x + alpha * y
    """
    # Input validation
    assert x.is_cuda and y.is_cuda, "Input tensors must be on CUDA device"
    assert x.shape == y.shape, "Input tensors must have the same shape"
    
    # Ensure contiguous memory layout for optimal access
    x_contig = x.contiguous()
    y_contig = y.contiguous()
    
    # Flatten tensors for 1D processing
    x_flat = x_contig.flatten()
    y_flat = y_contig.flatten()
    n_elements = x_flat.numel()
    
    # Allocate contiguous output
    output = torch.empty_like(x_flat)
    
    # Launch auto-tuned kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    add_kernel_optimized[grid](
        x_flat, y_flat, output, n_elements, alpha
    )
    
    # Reshape back to original shape
    return output.reshape(x.shape)
