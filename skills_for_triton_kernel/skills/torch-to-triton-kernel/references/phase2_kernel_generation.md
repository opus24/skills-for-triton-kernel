# Phase 2: 4가지 Triton Kernel 작성

## 목표
2가지 최적화 기법(A, B)을 **Phase 1에서 op별로 선정**하고, 그 조합으로 4개 커널 변형 생성

## 커널 변형

**Phase 1에서 선정한 A, B**를 각 커널에 맞게 적용한다.

### v1_baseline.py
- **최적화**: 없음
- **특징**: torch operation을 직접적으로 1:1 변환
- **용도**: 성능 비교의 기준선

### v2_opt_a.py
- **최적화**: Phase 1에서 선정한 **최적화 A만** 적용
- **특징**: A가 Tiling, Tensor Core 등이면 BLOCK_SIZE 활용; A가 Online, Reduction 등이면 해당 알고리즘 구현

### v3_opt_b.py
- **최적화**: Phase 1에서 선정한 **최적화 B만** 적용
- **특징**: B에 따라 메모리 접근(Coalescing), 리덕션, fusion 등 적용

### v4_opt_ab.py
- **최적화**: **A + B** 둘 다 적용
- **특징**: A·B 각각의 요구사항을 모두 만족

## 작성 규칙

### 1. 파일 위치
모든 커널 파일은 다음 위치에 생성:
```
kernels/{op_name}/v1_baseline.py
kernels/{op_name}/v2_opt_a.py
kernels/{op_name}/v3_opt_b.py
kernels/{op_name}/v4_opt_ab.py
```

예시: `kernels/softmax/v2_opt_a.py` (softmax의 A=Online Algorithm이면, 이 파일에는 Online만 적용)

### 2. 커널 구조

기본 구조는 다음과 같습니다:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def kernel_name(
    input_ptr,
    output_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ...
):
    # Triton kernel 구현
    pid = tl.program_id(0)
    # ... kernel logic ...
    pass

def triton_op(input_tensor, ...):
    """
    호출 래퍼 함수
    torch와 동일한 인터페이스 제공
    """
    # 입력 검증
    # 메모리 할당
    # 커널 실행
    # 결과 반환
    pass
```

### 3. 필수 요소

각 커널은 다음을 포함해야 합니다:

- **`@triton.jit` 데코레이터**: 커널 함수에 필수
- **Phase 1에서 선정한 A, B**를 각 커널에 맞게 적용:
  - **v2_opt_a**: A가 Tiling, Tensor Core 등이면 BLOCK_SIZE 사용; A가 Online, Reduction 등이면 해당 기법의 요구사항 적용
  - **v3_opt_b**: B에 따라 메모리 접근(Coalescing), 리덕션, fusion 등 적용
  - **v4_opt_ab**: A·B 각각의 요구사항을 모두 만족
- **torch와 동일한 입력/출력 인터페이스**: 모든 커널에 필수
- **에러 처리**: 입력 검증 및 예외 처리

### 4. 튜닝 파라미터

커널 성능 최적화를 위한 주요 파라미터. **A 또는 B가 Tiling, Tensor Core 등 block 사용 기법이면 BLOCK_SIZE 등이 해당 커널에 필수**이다.

- **`BLOCK_SIZE`**: 16, 32, 64, 128 중 선택
  - 작은 값: 더 많은 병렬성, 더 많은 메모리 접근
  - 큰 값: 더 적은 메모리 접근, 더 적은 병렬성
  - 일반적으로 64 또는 128이 좋은 시작점

- **`num_warps`**: 워프 수 (보통 4, 8, 16)
  - 더 많은 워프: 더 많은 병렬성, 더 많은 레지스터 사용
  - 일반적으로 4 또는 8이 적절

- **`num_stages`**: 파이프라인 스테이지 수 (선택적)
  - 더 많은 스테이지: 더 많은 메모리 사용, 더 나은 성능
  - 기본값 사용 권장

### 5. 코드 품질

- **명확한 변수명**: 의미 있는 이름 사용
- **주석**: 복잡한 로직에 설명 추가
- **일관성**: 다른 커널과 유사한 구조 유지
- **독립성**: 각 커널은 독립적으로 실행 가능해야 함

## 출력
- `kernels/{op_name}/` 디렉토리에 4개 파일 생성
- 각 커널은 독립적으로 실행 가능해야 함
- 모든 커널이 동일한 인터페이스를 제공해야 함

## 체크리스트
- [ ] v1_baseline.py 생성 완료
- [ ] v2_opt_a.py 생성 완료 (최적화 A 적용)
- [ ] v3_opt_b.py 생성 완료 (최적화 B 적용)
- [ ] v4_opt_ab.py 생성 완료 (A+B 적용)
- [ ] 모든 커널이 torch와 동일한 인터페이스 제공
- [ ] 모든 커널이 독립적으로 실행 가능
- [ ] 적절한 BLOCK_SIZE 및 튜닝 파라미터 설정 (A 또는 B가 해당 시)
