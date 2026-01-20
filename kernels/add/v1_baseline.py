"""
v1_baseline: Add Triton Kernel (No Optimization)
직접적인 1:1 변환, 최적화 없음
"""
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Baseline add kernel without optimization.
    Computes output = x + alpha * y
    
    Args:
        x_ptr: First input tensor pointer
        y_ptr: Second input tensor pointer
        output_ptr: Output tensor pointer
        n_elements: Total number of elements
        alpha: Scalar multiplier for y
        BLOCK_SIZE: Block size for processing
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Compute add: x + alpha * y
    output = x + alpha * y
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Triton add implementation (baseline).
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
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    add_kernel[grid](
        x_flat, y_flat, output, n_elements, alpha,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape back to original shape
    return output.reshape(x.shape)
