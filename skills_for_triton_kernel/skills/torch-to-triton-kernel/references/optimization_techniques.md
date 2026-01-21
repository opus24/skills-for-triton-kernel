# 최적화 기법 상세 설명

## 1. Tiling

### 설명
데이터를 BLOCK_SIZE 단위 타일로 분할하여 공유 메모리에서 처리하는 기법입니다.

### 효과
- 캐시 효율성 극대화
- L2 캐시 재사용률 향상
- 메모리 접근 패턴 최적화

### 적용 방법
- 입력 텐서를 BLOCK_SIZE x BLOCK_SIZE 타일로 분할
- 각 타일을 순차적으로 처리하여 메모리 접근 최소화
- BLOCK_SIZE는 보통 16, 32, 64, 128 중 선택
- 타일 단위로 처리하여 캐시 지역성 향상

### 구현 예시
```python
BLOCK_SIZE = 64

# 타일 단위로 데이터 로드
for i in range(0, N, BLOCK_SIZE):
    for j in range(0, M, BLOCK_SIZE):
        # 타일 데이터 처리
        tile = input[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
        process_tile(tile)
```

## 2. Memory Coalescing

### 설명
연속 메모리 접근 패턴으로 글로벌 메모리 로드/스토어를 최적화하는 기법입니다.

### 효과
- 메모리 대역폭 활용률 극대화
- 메모리 접근 지연 시간 감소
- GPU 메모리 버스 효율성 향상

### 적용 방법
- 연속된 메모리 주소에 대한 접근을 그룹화
- 벡터화된 로드/스토어 연산 활용 (tl.load, tl.store)
- 스레드 간 메모리 접근 패턴 정렬
- 메모리 접근을 연속적인 블록으로 구성

### 구현 예시
```python
# 연속 메모리 접근 패턴
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
# 연속된 메모리 주소에 대한 벡터화된 로드
data = tl.load(input_ptr + offsets)
```

## 3. Online Algorithm

### 설명
스트리밍 방식으로 블록을 순회하며 running max, running sum(exp) 등을 갱신하는 기법입니다. 두 번의 패스(max 찾기, exp/sum 계산)를 한 번에 처리합니다.

### 효과
- 수치 안정성 향상 (큰 값에서 overflow/underflow 완화)
- 메모리 패스 수 감소
- 긴 축(행/열)에서도 단일 패스로 처리 가능

### 적용 방법
- 블록별로 `m_new = max(m_i, max(block))`, `l_i = l_i * exp(m_i - m_new) + sum(exp(block - m_new))` 형태로 갱신
- softmax, log_softmax 등 log-sum-exp 계열에 적용

### 구현 요약
```python
# Online softmax: 블록별 running max, running sum of exp 갱신
m_i, l_i = -inf, 0.0
for block_start in range(0, n_cols, BLOCK_SIZE):
    x = tl.load(...)
    m_new = tl.maximum(m_i, tl.max(x, axis=0))
    l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
    m_i = m_new
# 두 번째 루프: output = exp(x - m_i) / l_i
```

## 4. Reduction

### 설명
블록 내부에서 `tl.sum`, `tl.max` 등으로 1차 리덕션을 수행한 뒤, 필요 시 블록 간 결과를 병합하는 tree reduction 기법입니다.

### 효과
- 메모리 대역폭 절감
- warp/block 단위 병렬 리덕션 활용
- 글로벌 메모리 쓰기 횟수 감소

### 적용 방법
- 블록 내 `block_sum = tl.sum(x, axis=0)` 또는 `block_max = tl.max(x, axis=0)`
- 다단계 리덕션이 필요하면 중간 결과를 저장하고 2차 커널 또는 동일 커널 내에서 병합
- layernorm: mean/var 계산에 적용

### 구현 요약
```python
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
block_sum = tl.sum(x, axis=0)
tl.store(out_ptr + pid, block_sum)  # 블록별 부분합
# 필요 시 2차 커널에서 블록 결과 병합
```

## 5. Tensor Core

### 설명
`tl.dot()`를 사용해 행렬 곱을 Tensor Core 하드웨어로 가속하는 기법입니다. BLOCK 차원을 16 배수로 맞추고, FP16/BF16/TF32 등을 사용합니다.

### 효과
- 행렬 곱 연산량 대비 throughput 극대화
- 메모리 대역폭 활용 효율 향상

### 적용 방법
- `acc += tl.dot(a_tile, b_tile)` 사용. `a_tile`, `b_tile` shape는 16 배수 (예: BLOCK_M, BLOCK_K, BLOCK_N)
- GEMM 루프 내 K 차원 타일링과 결합
- matmul, bmm, mm 등에 적용

### 구현 요약
```python
# BLOCK_M, BLOCK_N, BLOCK_K는 16 배수
a_tile = tl.load(a_ptr + ...)   # [BLOCK_M, BLOCK_K]
b_tile = tl.load(b_ptr + ...)   # [BLOCK_K, BLOCK_N]
acc += tl.dot(a_tile, b_tile)
```

## 6. Kernel Fusion

### 설명
여러 연산(예: matmul+bias+activation, mean+var+layernorm)을 하나의 커널로 합쳐 중간 결과를 글로벌 메모리에 쓰지 않는 기법입니다.

### 효과
- 메모리 트래픽 감소
- 커널 launch 오버헤드 감소
- 캐시/레지스터 재사용 증가

### 적용 방법
- 인접 op를 한 커널에서 순차 적용 (로드 → 연산1 → 연산2 → 스토어)
- element-wise(relu, gelu, silu)는 이전 op 출력을 바로 사용하도록 fusion
- layernorm은 mean/var 계산과 (x-mean)/sqrt(var+eps)를 한 커널에서 처리

### 구현 요약
```python
# 예: layernorm = mean/var + (x-mean)/sqrt(var+eps) 한 커널
# 예: gelu = x * 0.5 * (1 + tanh(...)) 이전 op 출력에서 바로 계산
```

---

## Op별 추천 최적화 기법 (A, B)

Phase 1에서 입력 op에 대해 **최적화 A**, **최적화 B** 두 가지를 이 표를 참고해 선정합니다. 표에 없는 op는 기본값 (Tiling, Memory Coalescing)을 사용하고, 유사 op(예: silu→gelu, bmm→matmul)를 참고해 조정할 수 있습니다.

| Operation | A | B | 비고 |
|-----------|---|---|------|
| softmax, log_softmax | Online Algorithm | Tiling | |
| layernorm | Reduction | Tiling | |
| gelu, silu, relu | Memory Coalescing | Kernel Fusion | element-wise, 이전 op와 fusion 가능 시 |
| add, sub, mul | Tiling | Memory Coalescing | |
| matmul, bmm, mm | Tiling | Tensor Core | GEMM |
| **미등록 op** | Tiling | Memory Coalescing | 기본값. Phase 1에서 유사 op 참고해 변경 가능 |

---

## 조합 전략

- **A, B 선정**: 위 "Op별 추천 최적화 기법 (A, B)" 표를 참고. 표에 없는 op는 (Tiling, Memory Coalescing) 기본 또는 Phase 1 조사로 유사 op를 참고해 결정.
- **4커널**:
  - **v1_baseline**: 최적화 없음, 1:1 변환, 기준선
  - **v2_opt_a**: 최적화 A만 적용
  - **v3_opt_b**: 최적화 B만 적용
  - **v4_opt_ab**: A + B 둘 다 적용
