"""
v2_tiling: Add Triton Kernel (Tiling Optimization)
BLOCK_SIZE 파라미터로 타일링 구현, 캐시 효율성 향상
"""
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel_tiled(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Tiled add kernel with optimized block processing.
    Computes output = x + alpha * y
    
    Tiling strategy:
    - Process data in BLOCK_SIZE chunks
    - Each program handles one tile
    - Improved cache locality within tiles
    
    Args:
        x_ptr: First input tensor pointer
        y_ptr: Second input tensor pointer
        output_ptr: Output tensor pointer
        n_elements: Total number of elements
        alpha: Scalar multiplier for y
        BLOCK_SIZE: Tile size for processing
    """
    pid = tl.program_id(0)
    
    # Calculate tile boundaries
    tile_start = pid * BLOCK_SIZE
    offsets = tile_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load tile data with boundary check
    x_tile = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_tile = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Process tile: x + alpha * y
    result_tile = x_tile + alpha * y_tile
    
    # Store tile result
    tl.store(output_ptr + offsets, result_tile, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def add_kernel_tiled_autotune(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Auto-tuned tiled add kernel.
    Automatically selects optimal BLOCK_SIZE based on input size.
    """
    pid = tl.program_id(0)
    
    tile_start = pid * BLOCK_SIZE
    offsets = tile_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x_tile = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_tile = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    result_tile = x_tile + alpha * y_tile
    
    tl.store(output_ptr + offsets, result_tile, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Triton add implementation with tiling optimization.
    Computes: x + alpha * y
    
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
    
    # Flatten tensors for processing
    x_flat = x.flatten()
    y_flat = y.flatten()
    n_elements = x_flat.numel()
    
    # Allocate output
    output = torch.empty_like(x_flat)
    
    # Use autotuned kernel for optimal performance
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    add_kernel_tiled_autotune[grid](
        x_flat, y_flat, output, n_elements, alpha
    )
    
    # Reshape back to original shape
    return output.reshape(x.shape)
