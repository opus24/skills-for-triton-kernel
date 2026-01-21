# Torch-to-Triton Kernel 예시

## kernels 디렉터리 구조

각 op마다 `kernels/{op_name}/` 아래에 4개 커널을 둡니다:

```
kernels/
├── add/
│   ├── __init__.py
│   ├── v1_baseline.py
│   ├── v2_tiling.py
│   ├── v3_coalesced.py
│   └── v4_optimized.py
├── softmax/
│   ├── __init__.py
│   ├── v1_baseline.py
│   ├── v2_tiling.py
│   ├── v3_coalesced.py
│   └── v4_optimized.py
└── mha/
    ├── __init__.py
    ├── v1_baseline.py
    ├── v2_tiling.py
    ├── v3_coalesced.py
    └── v4_optimized.py
```

## add: v1_baseline 예시

```python
"""
v1_baseline: Add Triton Kernel (No Optimization)
직접적인 1:1 변환, 최적화 없음
"""
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr, n_elements, alpha,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + alpha * y
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_add(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "Input tensors must be on CUDA device"
    assert x.shape == y.shape, "Input tensors must have the same shape"

    x_flat = x.flatten()
    y_flat = y.flatten()
    n_elements = x_flat.numel()
    output = torch.empty_like(x_flat)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x_flat, y_flat, output, n_elements, alpha, BLOCK_SIZE=BLOCK_SIZE)

    return output.reshape(x.shape)
```

## softmax, mha

- **softmax**: `kernels/softmax/v1_baseline.py` ~ `v4_optimized.py` — dim 축 기준 log-sum-exp, Tiling/Coalescing 조합.
- **mha**: `kernels/mha/v1_baseline.py` ~ `v4_optimized.py` — scaled dot-product attention, Q/K/V 포인터와 스케일 인자.

각 op의 `triton_*` 래퍼는 torch 동일 인터페이스(입력/출력 shape, dtype)를 유지합니다.

## 로그 파일 예시 (assets 스타일)

`logs/{op_name}_log.md`는 다음 형태를 따릅니다:

```markdown
# torch.add Triton Kernel 개발 로그

## Phase 1: 자료 조사 및 분석
- Operation: torch.add
- 수학적 정의: output = input + alpha * other
- 입력/출력: input, other (동일 shape), alpha=1.0
- 계산 복잡도: O(n), 메모리 바운드

## Phase 2: 커널 작성
### v1_baseline — BLOCK_SIZE: 1024, 최적화 없음
### v2_tiling — Tiling, Autotune
### v3_coalesced — Memory Coalescing
### v4_optimized — Tiling + Coalescing + Autotune

## Phase 3: Correctness Check
- v1~v4: ✅ 통과 (재시도 0~1회)

## Phase 3: 벤치마크 결과
| Kernel | Small | Medium | Large | Final | Speedup |
| v1_baseline | ... | ... | ... | ... | ...x |
| v2_tiling | ... | ... | ... | ... | ...x |
| v3_coalesced | ... | ... | ... | ... | ...x |
| v4_optimized | ... | ... | ... | ... | ...x |
| torch | ... | ... | ... | ... | 1.0x |

## Phase 4: 최적 커널
- 선택된 커널: v4_optimized
- Speedup: ...x
- 분석: Tiling+Coalescing 조합이 대부분 크기에서 우수
```

로그 템플릿은 `assets/log_template.md` 형식을 참고하여 Phase 1~4와 체크리스트를 채웁니다.
