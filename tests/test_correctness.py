"""
Correctness Test Suite for Triton Kernels
각 커널이 torch operation과 동일한 결과를 생성하는지 검증
"""
import torch
import numpy as np
from typing import Callable, Tuple, Dict, Any, Optional
import importlib
import sys
import os


def generate_test_cases(op_name: str, num_cases: int = 5) -> list:
    """
    Generate test cases for a given operation.
    
    Args:
        op_name: Name of the operation (e.g., 'softmax')
        num_cases: Number of test cases to generate
    
    Returns:
        List of test inputs
    """
    test_cases = []
    
    # Common test shapes
    shapes = [
        (32, 64),
        (128, 256),
        (256, 512),
        (1024, 2048),
        (1, 100),
    ]
    
    for i, shape in enumerate(shapes[:num_cases]):
        # Generate random input
        x = torch.randn(*shape, dtype=torch.float32, device='cuda')
        
        # Add edge cases
        if i == 0:
            # Normal case
            test_cases.append((x,))
        elif i == 1:
            # Small values
            test_cases.append((x * 0.1,))
        elif i == 2:
            # Large values
            test_cases.append((x * 10.0,))
        elif i == 3:
            # Negative values
            test_cases.append((x - 5.0,))
        else:
            # Edge case: single element
            test_cases.append((torch.randn(1, 1, dtype=torch.float32, device='cuda'),))
    
    return test_cases


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
        # Try to import from torch
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
        # Find the triton function (usually named triton_<op_name> or similar)
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


def run_correctness_test(
    op_name: str,
    variant: str,
    torch_op: Callable,
    triton_op: Callable,
    test_cases: list,
    atol: float = 1e-5,
    rtol: float = 1e-5
) -> Dict[str, Any]:
    """
    Run correctness test for a kernel variant.
    
    Args:
        op_name: Name of the operation
        variant: Kernel variant name
        torch_op: Torch operation function
        triton_op: Triton kernel function
        test_cases: List of test input tuples
        atol: Absolute tolerance
        rtol: Relative tolerance
    
    Returns:
        Dictionary with test results
    """
    results = {
        'passed': True,
        'errors': [],
        'test_results': []
    }
    
    for i, test_input in enumerate(test_cases):
        try:
            # Run torch operation
            torch_output = torch_op(*test_input)
            
            # Run triton operation
            triton_output = triton_op(*test_input)
            
            # Compare results
            if not torch.allclose(triton_output, torch_output, atol=atol, rtol=rtol):
                error_info = {
                    'test_case': i,
                    'max_diff': torch.max(torch.abs(triton_output - torch_output)).item(),
                    'mean_diff': torch.mean(torch.abs(triton_output - torch_output)).item(),
                    'torch_shape': torch_output.shape,
                    'triton_shape': triton_output.shape,
                }
                results['errors'].append(error_info)
                results['passed'] = False
                results['test_results'].append({
                    'test_case': i,
                    'passed': False,
                    'error': error_info
                })
            else:
                results['test_results'].append({
                    'test_case': i,
                    'passed': True
                })
        
        except Exception as e:
            error_info = {
                'test_case': i,
                'exception': str(e),
                'exception_type': type(e).__name__
            }
            results['errors'].append(error_info)
            results['passed'] = False
            results['test_results'].append({
                'test_case': i,
                'passed': False,
                'error': error_info
            })
    
    return results


def test_kernel_correctness(
    op_name: str,
    max_retries: int = 3,
    atol: float = 1e-5,
    rtol: float = 1e-5
) -> Dict[str, Any]:
    """
    Test correctness for all kernel variants with retry logic.
    
    Args:
        op_name: Name of the operation
        max_retries: Maximum number of regeneration attempts
        atol: Absolute tolerance
        rtol: Relative tolerance
    
    Returns:
        Dictionary with results for all variants
    """
    variants = ['v1_baseline', 'v2_tiling', 'v3_coalesced', 'v4_optimized']
    torch_op = get_torch_op(op_name)
    test_cases = generate_test_cases(op_name)
    
    all_results = {}
    
    for variant in variants:
        retry_count = 0
        variant_results = {
            'variant': variant,
            'passed': False,
            'retry_count': 0,
            'final_result': None,
            'errors': []
        }
        
        while retry_count < max_retries:
            try:
                # Import triton kernel
                triton_op = get_triton_kernel(op_name, variant)
                
                # Run correctness test
                result = run_correctness_test(
                    op_name, variant, torch_op, triton_op,
                    test_cases, atol, rtol
                )
                
                if result['passed']:
                    variant_results['passed'] = True
                    variant_results['retry_count'] = retry_count
                    variant_results['final_result'] = result
                    break
                else:
                    retry_count += 1
                    variant_results['errors'].extend(result['errors'])
                    
                    if retry_count < max_retries:
                        # Log that regeneration is needed
                        print(f"⚠️  {op_name}/{variant} failed correctness test (attempt {retry_count}/{max_retries})")
                        print(f"   Errors: {result['errors']}")
                        # Note: Actual regeneration should be handled by the agent
                        # This function just reports the need for regeneration
                    
            except Exception as e:
                retry_count += 1
                variant_results['errors'].append({
                    'exception': str(e),
                    'exception_type': type(e).__name__,
                    'retry': retry_count
                })
                
                if retry_count < max_retries:
                    print(f"⚠️  {op_name}/{variant} raised exception (attempt {retry_count}/{max_retries}): {e}")
        
        if not variant_results['passed']:
            variant_results['final_result'] = {
                'passed': False,
                'errors': variant_results['errors']
            }
        
        all_results[variant] = variant_results
    
    return all_results


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python test_correctness.py <op_name>")
        sys.exit(1)
    
    op_name = sys.argv[1]
    results = test_kernel_correctness(op_name)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Correctness Test Results for {op_name}")
    print(f"{'='*60}")
    
    for variant, result in results.items():
        status = "✅ PASSED" if result['passed'] else "❌ FAILED"
        retries = result['retry_count']
        print(f"{variant}: {status} (retries: {retries})")
        if not result['passed']:
            print(f"  Errors: {len(result['errors'])}")
