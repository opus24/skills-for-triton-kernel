# torch.softmax Triton Kernel 개발 로그

## Phase 1: 자료 조사 및 분석

### Operation 정보
- **Operation 이름**: torch.nn.functional.softmax
- **정확한 import 경로**: `torch.nn.functional.softmax(input, dim=-1)`
- **수학적 정의**: `softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))`

### 입력/출력 사양
- **입력 파라미터**:
  - `input`: 입력 텐서 (Shape: arbitrary, Dtype: float32/float16/bfloat16)
  - `dim`: softmax를 적용할 차원 (default: -1)

- **출력**:
  - Shape: input과 동일
  - Dtype: input과 동일

### 계산 복잡도
- **시간 복잡도**: O(n) - n은 softmax 차원의 크기
- **공간 복잡도**: O(n) - 출력 텐서 저장
- **메모리 접근 패턴**: Row-wise, 각 row 독립적으로 처리

## Phase 2: 커널 작성

### v1_baseline
- **특징**: torch.softmax를 직접적으로 1:1 변환
- **최적화**: 없음
- **BLOCK_SIZE**: next_power_of_2(n_cols), 64-4096

### v2_tiling
- **특징**: Tiling 기법 적용 (3-pass algorithm)
- **최적화**: Tiling + Autotune
- **BLOCK_SIZE**: Autotune (256, 512, 1024, 2048)
- **특이사항**: 큰 row 처리 가능, 3번 메모리 접근 오버헤드

### v3_coalesced
- **특징**: Memory Coalescing 기법 적용
- **최적화**: Memory Coalescing (contiguous 보장)
- **BLOCK_SIZE**: next_power_of_2(n_cols), 64-4096

### v4_optimized
- **특징**: Tiling + Memory Coalescing 조합
- **최적화**: Tiling + Memory Coalescing + Autotune
- **BLOCK_SIZE**: Autotune (256-4096) 또는 single-pass (n_cols <= 4096)

## Phase 3: Correctness Check

각 커널의 correctness check 결과:

- **v1_baseline**: PASSED (재시도 0회)
- **v2_tiling**: PASSED (재시도 0회)
- **v3_coalesced**: PASSED (재시도 0회)
- **v4_optimized**: PASSED (재시도 0회)

### 상세 정보
- 테스트 Shape: (8, 64), (32, 128), (64, 256), (128, 512), (256, 1024)
- Tolerance: atol=1e-4, rtol=1e-4
- 모든 테스트 케이스에서 torch.softmax와 동일한 결과 확인

## Phase 3: 벤치마크 결과

### 성능 비교 테이블

| Kernel | Small (ms) | Medium (ms) | Large (ms) | Final (ms) | Speedup vs Torch |
|--------|------------|-------------|------------|------------|------------------|
| v1_baseline | 0.0062 | 0.0185 | 0.2428 | 0.0892 | 1.00x |
| v2_tiling | 0.0067 | 0.0191 | 0.4295 | 0.1518 | 0.59x |
| v3_coalesced | 0.0056 | 0.0179 | 0.2423 | 0.0886 | 1.00x |
| v4_optimized | 0.0055 | 0.0183 | 0.2422 | 0.0887 | 1.00x |
| torch (baseline) | 0.0065 | 0.0189 | 0.2410 | 0.0888 | 1.0x |

### 벤치마크 설정
- **Small 텐서**: (256, 256) - 10회 실행 평균
- **Medium 텐서**: (1024, 1024) - 10회 실행 평균
- **Large 텐서**: (4096, 4096) - 10회 실행 평균
- **최종 성능**: (Small 평균 + Medium 평균 + Large 평균) / 3
- **벤치마크 도구**: `triton.testing.do_bench` (warmup=25, rep=100)
- **GPU**: NVIDIA A40 (46GB)

## Phase 4: 최적 커널

### 선택된 커널
- **커널**: v3_coalesced
- **최종 성능**: 0.0886 ms
- **Speedup**: 1.00x vs torch

### 선정 이유
v3_coalesced가 최저 평균 실행 시간(0.0886 ms)을 기록하여 최적 커널로 선정되었습니다. v4_optimized와 거의 동일한 성능을 보이지만, v3_coalesced가 더 단순한 구현으로 동일 수준의 성능을 달성했습니다.

### 성능 분석
1. **torch와 동등한 성능**: softmax는 이미 torch에서 고도로 최적화되어 있어 Triton 커널과의 차이가 거의 없습니다.

2. **v2_tiling의 성능 저하**: 3-pass 알고리즘(max 계산, sum 계산, normalize)으로 인해 large 텐서에서 메모리 접근 오버헤드가 발생했습니다.

3. **Memory Coalescing 효과**: v3_coalesced와 v4_optimized가 small 텐서에서 약간의 성능 향상(1.15-1.17x)을 보였습니다.

### 개선 여지
1. **Fused Kernels**: softmax를 다른 연산(dropout, matmul)과 fusion하면 메모리 접근 횟수 감소
2. **Online Softmax**: FlashAttention 스타일의 online softmax로 single-pass 구현 가능
3. **Half Precision**: float16/bfloat16 지원으로 메모리 대역폭 효율 향상
