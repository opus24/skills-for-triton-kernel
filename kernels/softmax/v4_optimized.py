"""
v4_optimized: Softmax Triton Kernel with Tiling + Memory Coalescing
두 최적화 기법을 모두 적용한 최적화 버전
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
    key=['n_cols'],
)
@triton.jit
def softmax_kernel_optimized(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fully optimized softmax kernel combining tiling and memory coalescing.
    
    Optimizations:
    1. Tiling: Process large rows in chunks
    2. Memory Coalescing: Contiguous memory access patterns
    3. Auto-tuning: Dynamic BLOCK_SIZE selection
    4. Numerical stability: Uses max subtraction
    
    Args:
        output_ptr: Output tensor pointer
        input_ptr: Input tensor pointer
        input_row_stride: Stride between rows in input
        output_row_stride: Stride between rows in output
        n_cols: Number of columns
        BLOCK_SIZE: Optimized tile size (auto-tuned)
    """
    row_idx = tl.program_id(0)
    
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    
    # First pass: compute global max across all tiles (coalesced reads)
    row_max = -float('inf')
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        # Coalesced load
        block_data = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
        block_max = tl.max(block_data, axis=0)
        row_max = tl.maximum(row_max, block_max)
    
    # Second pass: compute sum of exp(x - max) (coalesced reads)
    row_sum = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        # Coalesced load
        block_data = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
        block_exp = tl.exp(block_data - row_max)
        block_exp = tl.where(mask, block_exp, 0.0)
        row_sum += tl.sum(block_exp, axis=0)
    
    # Third pass: normalize and store (coalesced read/write)
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        # Coalesced load
        block_data = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
        block_exp = tl.exp(block_data - row_max)
        softmax_output = block_exp / row_sum
        # Coalesced store
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)


@triton.jit
def softmax_kernel_optimized_single_pass(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized single-pass softmax for rows that fit in BLOCK_SIZE.
    Most efficient when entire row can be processed at once.
    """
    row_idx = tl.program_id(0)
    
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    
    # Coalesced load of entire row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # Fused max-exp-sum-normalize
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    # Coalesced store
    tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)


def triton_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Triton softmax implementation with full optimization.
    Combines tiling and memory coalescing for maximum performance.
    
    Args:
        x: Input tensor
        dim: Dimension along which to apply softmax (default: -1)
    
    Returns:
        Softmax output tensor
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    
    if dim < 0:
        dim = x.ndim + dim
    
    # Ensure contiguous layout for coalesced memory access
    if dim != x.ndim - 1:
        x = x.transpose(dim, -1).contiguous()
    else:
        x = x.contiguous()
    
    original_shape = x.shape
    n_cols = x.shape[-1]
    x_2d = x.view(-1, n_cols).contiguous()
    n_rows = x_2d.shape[0]
    
    output = torch.empty_like(x_2d)
    
    # Choose kernel based on row size
    if n_cols <= 4096:
        # Single-pass kernel for small rows
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        BLOCK_SIZE = max(BLOCK_SIZE, 64)
        softmax_kernel_optimized_single_pass[(n_rows,)](
            output,
            x_2d,
            x_2d.stride(0),
            output.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Auto-tuned multi-pass kernel for large rows
        softmax_kernel_optimized[(n_rows,)](
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
