# Torch-to-Triton Kernel Agent Skills

## 개요

이 skills 문서는 사용자가 입력한 여러 torch operations를 Triton kernel로 변환하는 agent의 워크플로우를 정의합니다. 각 operation에 대해 4단계 파이프라인을 순차적으로 실행하여 최적화된 커널을 생성하고 평가합니다.

## 핵심 요구사항

- **다중 Ops 처리**: 사용자가 여러 ops를 한번에 입력 가능 (예: "softmax, layernorm, gelu"). 각 op은 Phase 1-4를 모두 거쳐야 함
- **벤치마크 방식**: Small/Medium/Large 텐서 각 10회 실행, 3가지 평균의 최종 평균을 최종 성능으로 사용
- **재생성 로직**: Correctness check 실패 시 해당 커널이 통과할 때까지 재생성

## 최적화 기법 (2개)

### 1. Tiling
- **설명**: 데이터를 BLOCK_SIZE 단위 타일로 분할하여 공유 메모리에서 처리
- **효과**: 캐시 효율성 극대화, L2 캐시 재사용률 향상
- **적용 방법**: 
  - 입력 텐서를 BLOCK_SIZE x BLOCK_SIZE 타일로 분할
  - 각 타일을 순차적으로 처리하여 메모리 접근 최소화
  - BLOCK_SIZE는 보통 16, 32, 64, 128 중 선택

### 2. Memory Coalescing
- **설명**: 연속 메모리 접근 패턴으로 글로벌 메모리 로드/스토어 최적화
- **효과**: 메모리 대역폭 활용률 극대화
- **적용 방법**:
  - 연속된 메모리 주소에 대한 접근을 그룹화
  - 벡터화된 로드/스토어 연산 활용 (tl.load, tl.store)
  - 스레드 간 메모리 접근 패턴 정렬

## 전체 워크플로우

```
사용자 입력: ops 리스트 (예: "softmax, layernorm, gelu")
  ↓
각 op에 대해 순차적으로:
  Phase 1: 자료 조사 및 분석
  Phase 2: 4개 커널 생성 (v1_baseline, v2_tiling, v3_coalesced, v4_optimized)
  Phase 3: Correctness Check + 재생성 (필요시) → 벤치마크 실행
  Phase 4: 최적 커널 선정 및 로그 작성
  ↓
최종 요약 리포트 출력
```

## Phase 1: 자료 조사 및 분석

### 목표
입력된 torch operation에 대한 완전한 이해를 얻어 올바른 Triton kernel을 작성할 수 있도록 함

### 수행 작업

1. **Operation 식별 및 파싱**
   - 사용자 입력에서 operation 이름 추출 (예: "softmax", "torch.nn.functional.softmax", "F.softmax")
   - torch 모듈에서 해당 operation의 정확한 위치 확인

2. **공식 문서 조사**
   - PyTorch 공식 문서에서 operation의 API, 파라미터, 동작 방식 확인
   - 수학적 정의 및 알고리즘 설명 파악

3. **소스 코드 분석**
   - torch의 C++/Python 구현 코드 검토 (가능한 경우)
   - 입력/출력 텐서의 shape, dtype, 메모리 레이아웃 파악
   - 연산의 계산 복잡도 및 메모리 접근 패턴 분석

4. **정리 및 문서화**
   - 다음 정보를 정리:
     - Operation 이름 및 정확한 import 경로
     - 입력 파라미터 (shape, dtype, 기타 옵션)
     - 출력 shape 및 dtype
     - 수학적 정의 (공식)
     - 계산 복잡도 (시간/공간)
     - 메모리 접근 패턴

### 출력
- `logs/{op_name}_log.md`에 Phase 1 섹션 작성
- 분석 결과를 구조화된 형태로 기록

## Phase 2: 4가지 Triton Kernel 작성

### 목표
2가지 최적화 기법의 조합으로 총 4개의 커널 변형 생성

### 커널 변형

#### v1_baseline.py
- **최적화**: 없음
- **특징**: torch operation을 직접적으로 1:1 변환
- **용도**: 성능 비교의 기준선

#### v2_tiling.py
- **최적화**: Tiling만 적용
- **특징**: 
  - BLOCK_SIZE 파라미터로 타일링 구현
  - 데이터를 타일 단위로 분할하여 처리
  - 캐시 효율성 향상

#### v3_coalesced.py
- **최적화**: Memory Coalescing만 적용
- **특징**:
  - 연속 메모리 접근 패턴 최적화
  - 벡터화된 로드/스토어 활용
  - 메모리 대역폭 최대화

#### v4_optimized.py
- **최적화**: Tiling + Memory Coalescing 조합
- **특징**: 두 기법을 모두 적용하여 최대 성능 추구

### 작성 규칙

1. **파일 위치**: `kernels/{op_name}/v{1-4}_{variant_name}.py`

2. **커널 구조**:
   ```python
   import torch
   import triton
   import triton.language as tl
   
   @triton.jit
   def kernel_name(...):
       # Triton kernel 구현
       pass
   
   def triton_op(input_tensor, ...):
       # 호출 래퍼 함수
       # torch와 동일한 인터페이스 제공
       pass
   ```

3. **필수 요소**:
   - `@triton.jit` 데코레이터로 커널 정의
   - 적절한 BLOCK_SIZE 설정 (v2, v4)
   - 메모리 접근 최적화 (v3, v4)
   - torch와 동일한 입력/출력 인터페이스

4. **튜닝 파라미터**:
   - `BLOCK_SIZE`: 16, 32, 64, 128 중 선택
   - `num_warps`: 워프 수 (보통 4, 8, 16)
   - 필요시 `num_stages` 조정

### 출력
- `kernels/{op_name}/` 디렉토리에 4개 파일 생성
- 각 커널은 독립적으로 실행 가능해야 함

## Phase 3: Correctness Check 및 벤치마크

### 3-1. Correctness Check

#### 목표
생성된 커널이 torch operation과 동일한 결과를 생성하는지 검증

#### 검증 프로세스

```python
for each kernel in [v1_baseline, v2_tiling, v3_coalesced, v4_optimized]:
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        # 테스트 케이스 생성
        test_inputs = generate_test_cases(op_name)
        
        # torch 결과 계산
        torch_output = torch_op(*test_inputs)
        
        # triton 결과 계산
        try:
            triton_output = triton_op(*test_inputs)
            
            # 비교
            if torch.allclose(torch_output, triton_output, atol=1e-5, rtol=1e-5):
                # 통과
                break
            else:
                # 실패 - 에러 정보 수집
                error_info = collect_error_info(torch_output, triton_output)
                retry_count += 1
                if retry_count < max_retries:
                    # 커널 재생성
                    regenerate_kernel(kernel, error_info)
        except Exception as e:
            # 예외 발생 - 에러 정보 수집
            error_info = str(e)
            retry_count += 1
            if retry_count < max_retries:
                regenerate_kernel(kernel, error_info)
    
    if retry_count >= max_retries:
        # 최종 실패 보고
        log_failure(kernel, error_info)
```

#### 검증 기준
- `torch.allclose(triton_output, torch_output, atol=1e-5, rtol=1e-5)` 사용
- 다양한 shape, dtype 조합으로 테스트
- Edge case 포함 (빈 텐서, 단일 요소 등)

#### 재생성 로직
- Correctness check 실패 시:
  1. 에러 정보 분석 (shape mismatch, dtype 문제, 수치 오차 등)
  2. 문제 원인 파악
  3. 커널 코드 수정 또는 재작성
  4. 최대 3회 재시도

### 3-2. 벤치마크 실행

#### 목표
4개 커널의 성능을 측정하여 최적 커널 선정

#### 벤치마크 프로토콜

```python
TENSOR_SIZES = {
    "small": (256, 256),
    "medium": (1024, 1024),
    "large": (4096, 4096)
}
NUM_RUNS = 10

for kernel in [v1_baseline, v2_tiling, v3_coalesced, v4_optimized]:
    size_averages = []
    
    for size_name, size in TENSOR_SIZES.items():
        # 해당 크기에서 10회 실행
        run_times = []
        for _ in range(NUM_RUNS):
            # 테스트 입력 생성
            test_input = create_test_input(op_name, size)
            
            # 벤치마크 실행
            time_ms = triton.testing.do_bench(
                lambda: triton_op(*test_input),
                warmup=25,
                rep=100
            )
            run_times.append(time_ms)
        
        # 해당 크기의 평균 계산
        size_avg = sum(run_times) / len(run_times)
        size_averages.append(size_avg)
    
    # 최종 성능 = 3가지 크기 평균의 평균
    final_performance = sum(size_averages) / len(size_averages)
    kernel_results[kernel] = {
        "small": size_averages[0],
        "medium": size_averages[1],
        "large": size_averages[2],
        "final": final_performance
    }
```

#### 성능 측정 상세
- **Small 텐서**: (256, 256) - 10회 실행 후 평균
- **Medium 텐서**: (1024, 1024) - 10회 실행 후 평균
- **Large 텐서**: (4096, 4096) - 10회 실행 후 평균
- **최종 성능**: (Small 평균 + Medium 평균 + Large 평균) / 3
- `triton.testing.do_bench` 사용 (warmup=25, rep=100)

#### 비교 기준
- torch operation의 성능도 동일한 방식으로 측정
- Speedup = torch_time / triton_time 계산

### 출력
- 각 커널의 벤치마크 결과 (Small/Medium/Large/최종)
- torch 대비 speedup 계산
- `logs/{op_name}_log.md`에 결과 기록

## Phase 4: 최적 커널 선정 및 로그 작성

### 목표
벤치마크 결과를 분석하여 최적 커널을 선정하고 전체 프로세스를 문서화

### 수행 작업

1. **결과 비교 테이블 생성**
   ```
   | Kernel | Small (ms) | Medium (ms) | Large (ms) | Final (ms) | Speedup vs Torch |
   |--------|------------|-------------|------------|------------|------------------|
   | v1_baseline | ... | ... | ... | ... | ... |
   | v2_tiling | ... | ... | ... | ... | ... |
   | v3_coalesced | ... | ... | ... | ... | ... |
   | v4_optimized | ... | ... | ... | ... | ... |
   | torch (baseline) | ... | ... | ... | ... | 1.0x |
   ```

2. **최적 커널 선정**
   - 최종 성능이 가장 빠른 커널 선택
   - Speedup이 1.0x 이상인 경우 성공으로 간주

3. **프로세스 로그 작성**
   - `logs/{op_name}_log.md` 파일에 다음 내용 기록:
     - Phase 1: 자료 조사 결과
     - Phase 2: 커널 작성 과정 및 특징
     - Phase 3: Correctness check 결과 (재시도 이력 포함)
     - Phase 3: 벤치마크 결과 상세
     - Phase 4: 최적 커널 선정 및 분석

### 로그 파일 구조

```markdown
# {op_name} Triton Kernel 개발 로그

## Phase 1: 자료 조사 및 분석
- Operation: ...
- 수학적 정의: ...
- 입력/출력: ...
- 계산 복잡도: ...

## Phase 2: 커널 작성
### v1_baseline
- 특징: ...
- BLOCK_SIZE: ...

### v2_tiling
- 특징: ...
- BLOCK_SIZE: ...

### v3_coalesced
- 특징: ...
- 최적화 기법: ...

### v4_optimized
- 특징: ...
- 최적화 기법: ...

## Phase 3: Correctness Check
- v1_baseline: ✅ 통과 (재시도 0회)
- v2_tiling: ✅ 통과 (재시도 1회)
- v3_coalesced: ✅ 통과 (재시도 0회)
- v4_optimized: ✅ 통과 (재시도 0회)

## Phase 3: 벤치마크 결과
| Kernel | Small | Medium | Large | Final | Speedup |
|--------|-------|--------|-------|-------|---------|
| ... | ... | ... | ... | ... | ... |

## Phase 4: 최적 커널
- 선택된 커널: v4_optimized
- 최종 성능: ... ms
- Speedup: ...x
- 분석: ...
```

## 다중 Ops 처리

### 입력 형식
사용자는 쉼표로 구분된 여러 ops를 입력할 수 있습니다:
- 예: "softmax, layernorm, gelu"
- 예: "torch.nn.functional.softmax, torch.nn.functional.layer_norm"

### 처리 순서
각 op에 대해 **순차적으로** Phase 1-4를 모두 실행:
1. op1 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → 완료
2. op2 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → 완료
3. op3 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → 완료
4. 최종 요약 리포트 출력

### 최종 요약 리포트
모든 ops 처리 완료 후:
- 각 op별 최적 커널 및 성능 요약
- 전체 통계 (평균 speedup, 성공률 등)
- 생성된 파일 목록

## 구현 체크리스트

각 op 처리 시 다음을 확인:

- [ ] Phase 1: 자료 조사 완료 및 로그 작성
- [ ] Phase 2: 4개 커널 모두 생성 (v1-v4)
- [ ] Phase 3: 모든 커널이 correctness check 통과 (재시도 포함)
- [ ] Phase 3: 벤치마크 실행 (Small/Medium/Large 각 10회)
- [ ] Phase 4: 최적 커널 선정 및 최종 로그 작성
- [ ] 모든 파일이 올바른 위치에 생성됨

## 주의사항

1. **Correctness 우선**: 성능보다 정확도가 우선. Correctness check를 통과하지 못한 커널은 벤치마크하지 않음
2. **재시도 제한**: 최대 3회 재시도 후에도 실패하면 해당 커널은 스킵하고 다음으로 진행
3. **에러 처리**: 예외 발생 시 상세한 에러 메시지를 로그에 기록
4. **메모리 관리**: 큰 텐서 테스트 시 메모리 부족을 고려하여 적절한 크기 조정
