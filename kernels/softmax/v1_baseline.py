"""
v1_baseline: Softmax Triton Kernel (No Optimization)
직접적인 1:1 변환, 최적화 없음
"""
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel_baseline(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Baseline softmax kernel - processes one row per program.
    
    Args:
        output_ptr: Output tensor pointer
        input_ptr: Input tensor pointer
        input_row_stride: Stride between rows in input
        output_row_stride: Stride between rows in output
        n_cols: Number of columns (softmax dimension size)
        BLOCK_SIZE: Block size for processing columns
    """
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Compute row pointers
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    
    # Load entire row (assuming row fits in BLOCK_SIZE)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load row data
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability
    row_max = tl.max(row, axis=0)
    row_shifted = row - row_max
    
    # Compute exp
    numerator = tl.exp(row_shifted)
    
    # Compute sum
    denominator = tl.sum(numerator, axis=0)
    
    # Normalize
    softmax_output = numerator / denominator
    
    # Store result
    tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)


def triton_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Triton softmax implementation (baseline).
    
    Args:
        x: Input tensor
        dim: Dimension along which to apply softmax (default: -1)
    
    Returns:
        Softmax output tensor
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    
    # Handle negative dim
    if dim < 0:
        dim = x.ndim + dim
    
    # Move softmax dim to last position for easier processing
    if dim != x.ndim - 1:
        x = x.transpose(dim, -1).contiguous()
    else:
        x = x.contiguous()
    
    # Reshape to 2D: (batch, n_cols)
    original_shape = x.shape
    n_cols = x.shape[-1]
    x_2d = x.view(-1, n_cols)
    n_rows = x_2d.shape[0]
    
    # Allocate output
    output = torch.empty_like(x_2d)
    
    # Determine BLOCK_SIZE (must be power of 2 and >= n_cols)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = max(BLOCK_SIZE, 64)  # Minimum block size
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)  # Maximum block size
    
    # Launch kernel - one program per row
    softmax_kernel_baseline[(n_rows,)](
        output,
        x_2d,
        x_2d.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back
    output = output.view(original_shape)
    
    # Transpose back if needed
    if dim != len(original_shape) - 1:
        output = output.transpose(dim, -1)
    
    return output
