"""
v3_coalesced: Add Triton Kernel (Memory Coalescing Optimization)
연속 메모리 접근 패턴 최적화, 메모리 대역폭 최대화
"""
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel_coalesced(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Memory coalesced add kernel.
    Computes output = x + alpha * y
    
    Memory coalescing strategy:
    - Contiguous thread access pattern
    - Sequential memory addressing
    - Maximizes memory bandwidth utilization
    
    Args:
        x_ptr: First input tensor pointer (contiguous)
        y_ptr: Second input tensor pointer (contiguous)
        output_ptr: Output tensor pointer (contiguous)
        n_elements: Total number of elements
        alpha: Scalar multiplier for y
        BLOCK_SIZE: Number of elements per block
    """
    pid = tl.program_id(0)
    
    # Coalesced memory access pattern
    # Each thread block handles BLOCK_SIZE consecutive elements
    block_start = pid * BLOCK_SIZE
    
    # Thread indices within block for coalesced access
    thread_offsets = tl.arange(0, BLOCK_SIZE)
    global_offsets = block_start + thread_offsets
    
    # Boundary mask for safe memory access
    mask = global_offsets < n_elements
    
    # Coalesced loads - consecutive threads access consecutive memory
    x_data = tl.load(x_ptr + global_offsets, mask=mask, other=0.0)
    y_data = tl.load(y_ptr + global_offsets, mask=mask, other=0.0)
    
    # Compute element-wise addition
    result = x_data + alpha * y_data
    
    # Coalesced store - consecutive threads write to consecutive memory
    tl.store(output_ptr + global_offsets, result, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Triton add implementation with memory coalescing optimization.
    Computes: x + alpha * y
    
    This implementation ensures:
    - Input tensors are contiguous in memory
    - Memory access patterns are coalesced for maximum bandwidth
    
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
    
    # Ensure contiguous memory layout for coalesced access
    x_contig = x.contiguous()
    y_contig = y.contiguous()
    
    # Flatten tensors for 1D processing
    x_flat = x_contig.flatten()
    y_flat = y_contig.flatten()
    n_elements = x_flat.numel()
    
    # Allocate contiguous output
    output = torch.empty_like(x_flat)
    
    # Launch kernel with coalesced access pattern
    BLOCK_SIZE = 1024  # Power of 2 for optimal memory alignment
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    add_kernel_coalesced[grid](
        x_flat, y_flat, output, n_elements, alpha,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape back to original shape
    return output.reshape(x.shape)
