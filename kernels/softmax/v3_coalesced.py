"""
v3_coalesced: Softmax Triton Kernel with Memory Coalescing
Memory Coalescing 기법을 적용하여 메모리 대역폭 활용률 향상
"""
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel_coalesced(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Softmax kernel with memory coalescing optimization.
    Ensures contiguous memory access patterns for better bandwidth.
    
    Args:
        output_ptr: Output tensor pointer
        input_ptr: Input tensor pointer
        input_row_stride: Stride between rows in input
        output_row_stride: Stride between rows in output
        n_cols: Number of columns
        BLOCK_SIZE: Block size for coalesced access
    """
    row_idx = tl.program_id(0)
    
    # Compute contiguous row pointers
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    
    # Coalesced column offsets (consecutive threads access consecutive memory)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Coalesced load - threads access consecutive memory locations
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
    
    # Coalesced store - threads write to consecutive memory locations
    tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)


def triton_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Triton softmax implementation with memory coalescing.
    Ensures input is contiguous for optimal memory access patterns.
    
    Args:
        x: Input tensor
        dim: Dimension along which to apply softmax (default: -1)
    
    Returns:
        Softmax output tensor
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    
    if dim < 0:
        dim = x.ndim + dim
    
    # Ensure contiguous layout for coalesced access
    if dim != x.ndim - 1:
        x = x.transpose(dim, -1).contiguous()
    else:
        x = x.contiguous()
    
    original_shape = x.shape
    n_cols = x.shape[-1]
    x_2d = x.view(-1, n_cols).contiguous()  # Ensure row-major contiguous
    n_rows = x_2d.shape[0]
    
    # Allocate contiguous output
    output = torch.empty_like(x_2d)
    
    # Determine BLOCK_SIZE (power of 2 for optimal coalescing)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = max(BLOCK_SIZE, 64)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    
    # Launch kernel
    softmax_kernel_coalesced[(n_rows,)](
        output,
        x_2d,
        x_2d.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    output = output.view(original_shape)
    
    if dim != len(original_shape) - 1:
        output = output.transpose(dim, -1)
    
    return output
