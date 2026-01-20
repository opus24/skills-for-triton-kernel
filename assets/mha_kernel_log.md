# Multi-Head Attention Triton Kernel 개발 로그

## Phase 1: 자료 조사 및 분석

### Operation 정보
- **Operation 이름**: Multi-Head Attention (Scaled Dot-Product Attention)
- **정확한 import 경로**: `torch.nn.functional.scaled_dot_product_attention(query, key, value)`
- **수학적 정의**: `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V`

### 입력/출력 사양
- **입력 파라미터**:
  - `query`: Query 텐서 [batch, heads, seq_len, head_dim]
  - `key`: Key 텐서 [batch, heads, seq_len, head_dim]
  - `value`: Value 텐서 [batch, heads, seq_len, head_dim]
  - `causal`: Causal mask 적용 여부 (optional)

- **출력**:
  - Shape: [batch, heads, seq_len, head_dim]
  - Dtype: input과 동일

### 계산 복잡도
- **시간 복잡도**: O(n²d) - n은 sequence length, d는 head dimension
- **공간 복잡도**: O(n²) (naive), O(n) (Flash Attention)
- **메모리 접근 패턴**: Matrix multiplication + softmax, 메모리 바운드 연산

## Phase 2: 커널 작성

### v1_baseline
- **특징**: 기본 MHA 구현, online softmax 사용
- **최적화**: 없음
- **BLOCK_M/N/K**: 64/64/64 (고정)

### v2_tiling
- **특징**: Flash Attention 스타일 tiling
- **최적화**: Tiling + Autotune
- **BLOCK_M/N**: Autotune (32-128), BLOCK_K: head_dim 기반

### v3_coalesced
- **특징**: Memory Coalescing 기법 적용
- **최적화**: Memory Coalescing (contiguous access)
- **BLOCK_M/N/K**: 64/64/64 (고정)

### v4_optimized
- **특징**: Flash Attention 스타일 + Causal mask 지원
- **최적화**: Tiling + Memory Coalescing + Autotune + Pipeline stages
- **BLOCK_M/N**: Autotune (32-128), BLOCK_K: head_dim 기반

## Phase 3: Correctness Check

각 커널의 correctness check 결과:

- **v1_baseline**: PASSED (재시도 0회)
- **v2_tiling**: PASSED (재시도 0회)
- **v3_coalesced**: PASSED (재시도 0회)
- **v4_optimized**: PASSED (재시도 0회)

### 상세 정보
- 테스트 Shape: [1,4,32,32], [2,4,64,32], [2,8,64,64], [4,8,128,64]
- Tolerance: atol=1e-2, rtol=1e-2 (MHA의 복잡한 연산으로 인한 누적 오차 고려)
- 모든 테스트 케이스에서 torch SDPA와 동일한 결과 확인

## Phase 3: 벤치마크 결과

### 성능 비교 테이블

| Kernel | Small (ms) | Medium (ms) | Large (ms) | Final (ms) | Speedup vs Torch |
|--------|------------|-------------|------------|------------|------------------|
| v1_baseline | 0.0107 | 0.0224 | 0.0609 | 0.0313 | 2.05x |
| v2_tiling | 0.0095 | 0.0215 | 0.0449 | 0.0253 | 2.54x |
| v3_coalesced | 0.0104 | 0.0221 | 0.0605 | 0.0310 | 2.07x |
| v4_optimized | 0.0098 | 0.0237 | 0.0474 | 0.0270 | 2.38x |
| torch.SDPA (baseline) | 0.0158 | 0.0480 | 0.1289 | 0.0643 | 1.0x |

### 벤치마크 설정
- **Small 텐서**: [2, 8, 64, 64] - 10회 실행 평균
- **Medium 텐서**: [2, 8, 256, 64] - 10회 실행 평균
- **Large 텐서**: [2, 8, 512, 64] - 10회 실행 평균
- **최종 성능**: (Small 평균 + Medium 평균 + Large 평균) / 3
- **벤치마크 도구**: `triton.testing.do_bench` (warmup=25, rep=100)
- **GPU**: NVIDIA A40 (46GB)

## Phase 4: 최적 커널

### 선택된 커널
- **커널**: v2_tiling
- **최종 성능**: 0.0253 ms
- **Speedup**: 2.54x vs torch

### 선정 이유
v2_tiling이 최저 평균 실행 시간(0.0253 ms)을 기록하며 torch SDPA 대비 2.54배 빠른 성능을 달성하여 최적 커널로 선정되었습니다.

### 성능 분석
1. **모든 Triton 커널이 torch보다 빠름**: Online softmax와 tiling 기법으로 메모리 효율성이 크게 향상되었습니다.

2. **v2_tiling의 우수한 성능**: Autotune이 각 입력 크기에 최적의 BLOCK_M/N을 선택하여 Large 텐서에서 2.87x speedup을 달성했습니다.

3. **Tiling의 효과**: Large 텐서에서 tiling 기법(v2, v4)이 고정 블록 크기(v1, v3)보다 현저히 빠릅니다.

4. **Flash Attention 스타일**: Online softmax로 O(n²) attention matrix를 메모리에 저장하지 않아 메모리 효율적입니다.

### 개선 여지
1. **Half Precision**: float16/bfloat16 지원으로 2배 더 빠른 성능 기대
2. **Variable Sequence Length**: 가변 길이 시퀀스 배치 처리 지원
3. **GQA/MQA**: Grouped Query Attention, Multi-Query Attention 지원
4. **KV Cache**: 추론 시 KV 캐시 활용 최적화
5. **더 큰 시퀀스**: 4K+ 시퀀스 길이에서의 메모리 최적화
