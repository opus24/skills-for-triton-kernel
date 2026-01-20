"""
v2_tiling: Softmax Triton Kernel with Tiling Optimization
Tiling 기법을 적용하여 큰 row도 처리 가능하도록 함
"""
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel_tiling(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Softmax kernel with tiling - handles rows larger than BLOCK_SIZE.
    Uses two-pass algorithm: first pass for max/sum, second for normalization.
    
    Args:
        output_ptr: Output tensor pointer
        input_ptr: Input tensor pointer
        input_row_stride: Stride between rows in input
        output_row_stride: Stride between rows in output
        n_cols: Number of columns
        BLOCK_SIZE: Tile size for processing
    """
    row_idx = tl.program_id(0)
    
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    
    # First pass: compute max across all tiles
    row_max = -float('inf')
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        block_data = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
        block_max = tl.max(block_data, axis=0)
        row_max = tl.maximum(row_max, block_max)
    
    # Second pass: compute sum of exp(x - max)
    row_sum = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        block_data = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
        block_exp = tl.exp(block_data - row_max)
        block_exp = tl.where(mask, block_exp, 0.0)
        row_sum += tl.sum(block_exp, axis=0)
    
    # Third pass: normalize and store
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        block_data = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
        block_exp = tl.exp(block_data - row_max)
        softmax_output = block_exp / row_sum
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_cols'],
)
@triton.jit
def softmax_kernel_tiling_autotune(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Auto-tuned version of tiled softmax kernel."""
    row_idx = tl.program_id(0)
    
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    
    # First pass: compute max
    row_max = -float('inf')
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        block_data = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
        block_max = tl.max(block_data, axis=0)
        row_max = tl.maximum(row_max, block_max)
    
    # Second pass: compute sum
    row_sum = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        block_data = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
        block_exp = tl.exp(block_data - row_max)
        block_exp = tl.where(mask, block_exp, 0.0)
        row_sum += tl.sum(block_exp, axis=0)
    
    # Third pass: normalize and store
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        block_data = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
        block_exp = tl.exp(block_data - row_max)
        softmax_output = block_exp / row_sum
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)


def triton_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Triton softmax implementation with tiling optimization.
    Can handle rows of any size by processing in tiles.
    
    Args:
        x: Input tensor
        dim: Dimension along which to apply softmax (default: -1)
    
    Returns:
        Softmax output tensor
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    
    if dim < 0:
        dim = x.ndim + dim
    
    if dim != x.ndim - 1:
        x = x.transpose(dim, -1).contiguous()
    else:
        x = x.contiguous()
    
    original_shape = x.shape
    n_cols = x.shape[-1]
    x_2d = x.view(-1, n_cols)
    n_rows = x_2d.shape[0]
    
    output = torch.empty_like(x_2d)
    
    # Use autotuned kernel
    softmax_kernel_tiling_autotune[(n_rows,)](
        output,
        x_2d,
        x_2d.stride(0),
        output.stride(0),
        n_cols,
    )
    
    output = output.view(original_shape)
    
    if dim != len(original_shape) - 1:
        output = output.transpose(dim, -1)
    
    return output
