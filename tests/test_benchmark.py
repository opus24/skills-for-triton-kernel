"""
Benchmark Test Suite for Triton Kernels
3가지 텐서 크기(Small/Medium/Large)에서 각 10회 실행하여 성능 측정
"""
import torch
import triton
import numpy as np
from typing import Callable, Dict, List, Tuple, Any
import importlib
import sys
import time


# 벤치마크 텐서 크기 정의
TENSOR_SIZES = {
    "small": (256, 256),
    "medium": (1024, 1024),
    "large": (4096, 4096)
}
NUM_RUNS = 10  # 각 크기당 실행 횟수


def get_torch_op(op_name: str) -> Callable:
    """
    Get the torch operation function.
    
    Args:
        op_name: Name of the operation
    
    Returns:
        Torch operation function
    """
    op_mapping = {
        'softmax': torch.nn.functional.softmax,
        'layernorm': torch.nn.functional.layer_norm,
        'gelu': torch.nn.functional.gelu,
        'relu': torch.nn.functional.relu,
        'sigmoid': torch.sigmoid,
        'tanh': torch.tanh,
    }
    
    if op_name.lower() in op_mapping:
        return op_mapping[op_name.lower()]
    else:
        try:
            if hasattr(torch, op_name):
                return getattr(torch, op_name)
            elif hasattr(torch.nn.functional, op_name):
                return getattr(torch.nn.functional, op_name)
            else:
                raise ValueError(f"Unknown operation: {op_name}")
        except Exception as e:
            raise ValueError(f"Cannot find operation {op_name}: {e}")


def get_triton_kernel(op_name: str, variant: str) -> Callable:
    """
    Import and return the triton kernel function.
    
    Args:
        op_name: Name of the operation
        variant: Kernel variant (v1_baseline, v2_tiling, v3_coalesced, v4_optimized)
    
    Returns:
        Triton kernel function
    """
    kernel_path = f"kernels.{op_name}.{variant}"
    
    try:
        module = importlib.import_module(kernel_path)
        # Find the triton function
        for attr_name in dir(module):
            if attr_name.startswith('triton_') or (hasattr(module, attr_name) and 
                                                   callable(getattr(module, attr_name)) and 
                                                   not attr_name.startswith('_')):
                func = getattr(module, attr_name)
                if callable(func) and not isinstance(func, type):
                    return func
        
        # Fallback: try common names
        if hasattr(module, 'triton_softmax'):
            return module.triton_softmax
        elif hasattr(module, 'triton_op'):
            return module.triton_op
        else:
            raise AttributeError(f"No triton function found in {kernel_path}")
    except ImportError as e:
        raise ImportError(f"Cannot import kernel {kernel_path}: {e}")


def create_test_input(op_name: str, size: Tuple[int, ...], device: str = 'cuda') -> Tuple[torch.Tensor, ...]:
    """
    Create test input for a given operation and size.
    
    Args:
        op_name: Name of the operation
        size: Tensor size tuple
        device: Device to create tensor on
    
    Returns:
        Tuple of input tensors
    """
    # Generate random input
    x = torch.randn(*size, dtype=torch.float32, device=device)
    
    # Operation-specific adjustments
    if op_name.lower() == 'layernorm':
        # LayerNorm needs normalized_shape
        return (x, size[-1:])
    elif op_name.lower() in ['softmax', 'gelu', 'relu', 'sigmoid', 'tanh']:
        return (x,)
    else:
        return (x,)


def benchmark_single_run(
    func: Callable,
    test_input: Tuple[torch.Tensor, ...],
    warmup: int = 25,
    rep: int = 100
) -> float:
    """
    Run a single benchmark using triton.testing.do_bench.
    
    Args:
        func: Function to benchmark
        test_input: Input arguments
        warmup: Number of warmup iterations
        rep: Number of measurement iterations
    
    Returns:
        Average time in milliseconds
    """
    try:
        # Create a lambda that captures the inputs
        def benchmark_func():
            return func(*test_input)
        
        # Use triton's benchmark utility
        time_ms = triton.testing.do_bench(
            benchmark_func,
            warmup=warmup,
            rep=rep
        )
        return time_ms
    except Exception as e:
        print(f"Error in benchmark: {e}")
        return float('inf')


def benchmark_kernel(
    op_name: str,
    variant: str,
    torch_op: Callable,
    triton_op: Callable
) -> Dict[str, Any]:
    """
    Benchmark a single kernel variant across all tensor sizes.
    
    Args:
        op_name: Name of the operation
        variant: Kernel variant name
        torch_op: Torch operation function
        triton_op: Triton kernel function
    
    Returns:
        Dictionary with benchmark results
    """
    results = {
        'variant': variant,
        'sizes': {},
        'final_performance': 0.0,
        'torch_baseline': {},
        'speedup': 0.0
    }
    
    size_averages = []
    torch_size_averages = []
    
    # Benchmark each size
    for size_name, size in TENSOR_SIZES.items():
        print(f"  Benchmarking {size_name} size {size}...")
        
        # Create test input
        test_input = create_test_input(op_name, size)
        
        # Benchmark triton kernel (10 runs)
        triton_times = []
        for run_idx in range(NUM_RUNS):
            time_ms = benchmark_single_run(triton_op, test_input)
            triton_times.append(time_ms)
        
        triton_avg = np.mean(triton_times)
        size_averages.append(triton_avg)
        results['sizes'][size_name] = {
            'times': triton_times,
            'mean': triton_avg,
            'std': np.std(triton_times),
            'min': np.min(triton_times),
            'max': np.max(triton_times)
        }
        
        # Benchmark torch baseline (10 runs)
        torch_times = []
        for run_idx in range(NUM_RUNS):
            time_ms = benchmark_single_run(torch_op, test_input)
            torch_times.append(time_ms)
        
        torch_avg = np.mean(torch_times)
        torch_size_averages.append(torch_avg)
        results['torch_baseline'][size_name] = {
            'times': torch_times,
            'mean': torch_avg,
            'std': np.std(torch_times),
            'min': np.min(torch_times),
            'max': np.max(torch_times)
        }
        
        # Calculate speedup for this size
        speedup = torch_avg / triton_avg if triton_avg > 0 else 0.0
        results['sizes'][size_name]['speedup'] = speedup
    
    # Calculate final performance: average of 3 size averages
    results['final_performance'] = np.mean(size_averages)
    torch_final = np.mean(torch_size_averages)
    results['torch_final'] = torch_final
    results['speedup'] = torch_final / results['final_performance'] if results['final_performance'] > 0 else 0.0
    
    return results


def benchmark_all_kernels(op_name: str) -> Dict[str, Any]:
    """
    Benchmark all kernel variants for an operation.
    
    Args:
        op_name: Name of the operation
    
    Returns:
        Dictionary with results for all variants
    """
    variants = ['v1_baseline', 'v2_tiling', 'v3_coalesced', 'v4_optimized']
    torch_op = get_torch_op(op_name)
    
    all_results = {}
    
    print(f"\n{'='*60}")
    print(f"Benchmarking {op_name}")
    print(f"{'='*60}")
    
    for variant in variants:
        print(f"\nBenchmarking {variant}...")
        try:
            triton_op = get_triton_kernel(op_name, variant)
            result = benchmark_kernel(op_name, variant, torch_op, triton_op)
            all_results[variant] = result
            
            print(f"  ✅ {variant} completed")
            print(f"     Final performance: {result['final_performance']:.4f} ms")
            print(f"     Speedup vs torch: {result['speedup']:.2f}x")
        
        except Exception as e:
            print(f"  ❌ {variant} failed: {e}")
            all_results[variant] = {
                'variant': variant,
                'error': str(e)
            }
    
    return all_results


def print_benchmark_summary(results: Dict[str, Any], op_name: str):
    """
    Print a formatted summary of benchmark results.
    
    Args:
        results: Benchmark results dictionary
        op_name: Name of the operation
    """
    print(f"\n{'='*60}")
    print(f"Benchmark Summary for {op_name}")
    print(f"{'='*60}\n")
    
    # Table header
    print(f"{'Kernel':<20} {'Small (ms)':<12} {'Medium (ms)':<12} {'Large (ms)':<12} {'Final (ms)':<12} {'Speedup':<10}")
    print("-" * 80)
    
    # Find best kernel
    best_kernel = None
    best_perf = float('inf')
    
    for variant, result in results.items():
        if 'error' in result:
            print(f"{variant:<20} {'ERROR':<12}")
            continue
        
        small = result['sizes']['small']['mean']
        medium = result['sizes']['medium']['mean']
        large = result['sizes']['large']['mean']
        final = result['final_performance']
        speedup = result['speedup']
        
        print(f"{variant:<20} {small:<12.4f} {medium:<12.4f} {large:<12.4f} {final:<12.4f} {speedup:<10.2f}x")
        
        if final < best_perf:
            best_perf = final
            best_kernel = variant
    
    print("-" * 80)
    if best_kernel:
        print(f"\n🏆 Best Kernel: {best_kernel} ({best_perf:.4f} ms)")
        print(f"   Speedup: {results[best_kernel]['speedup']:.2f}x vs torch")


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python test_benchmark.py <op_name>")
        sys.exit(1)
    
    op_name = sys.argv[1]
    results = benchmark_all_kernels(op_name)
    print_benchmark_summary(results, op_name)
