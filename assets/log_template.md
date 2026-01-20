# {op_name} Triton Kernel 개발 로그

## Phase 1: 자료 조사 및 분석

### Operation 정보
- **Operation 이름**: {operation_name}
- **정확한 import 경로**: {import_path}
- **수학적 정의**: {mathematical_definition}

### 입력/출력 사양
- **입력 파라미터**:
  - Shape: {input_shape}
  - Dtype: {input_dtype}
  - 기타 옵션: {other_options}

- **출력**:
  - Shape: {output_shape}
  - Dtype: {output_dtype}

### 계산 복잡도
- **시간 복잡도**: {time_complexity}
- **공간 복잡도**: {space_complexity}
- **메모리 접근 패턴**: {memory_access_pattern}

## Phase 2: 커널 작성

### v1_baseline
- **특징**: torch operation을 직접적으로 1:1 변환
- **최적화**: 없음
- **BLOCK_SIZE**: N/A

### v2_tiling
- **특징**: Tiling 기법 적용
- **최적화**: Tiling
- **BLOCK_SIZE**: {v2_block_size}

### v3_coalesced
- **특징**: Memory Coalescing 기법 적용
- **최적화**: Memory Coalescing
- **BLOCK_SIZE**: N/A

### v4_optimized
- **특징**: Tiling + Memory Coalescing 조합
- **최적화**: Tiling + Memory Coalescing
- **BLOCK_SIZE**: {v4_block_size}

## Phase 3: Correctness Check

각 커널의 correctness check 결과:

- **v1_baseline**: {v1_correctness_status} (재시도 {v1_retry_count}회)
- **v2_tiling**: {v2_correctness_status} (재시도 {v2_retry_count}회)
- **v3_coalesced**: {v3_correctness_status} (재시도 {v3_retry_count}회)
- **v4_optimized**: {v4_correctness_status} (재시도 {v4_retry_count}회)

### 상세 정보
{correctness_details}

## Phase 3: 벤치마크 결과

### 성능 비교 테이블

| Kernel | Small (ms) | Medium (ms) | Large (ms) | Final (ms) | Speedup vs Torch |
|--------|------------|-------------|------------|------------|------------------|
| v1_baseline | {v1_small} | {v1_medium} | {v1_large} | {v1_final} | {v1_speedup}x |
| v2_tiling | {v2_small} | {v2_medium} | {v2_large} | {v2_final} | {v2_speedup}x |
| v3_coalesced | {v3_small} | {v3_medium} | {v3_large} | {v3_final} | {v3_speedup}x |
| v4_optimized | {v4_small} | {v4_medium} | {v4_large} | {v4_final} | {v4_speedup}x |
| torch (baseline) | {torch_small} | {torch_medium} | {torch_large} | {torch_final} | 1.0x |

### 벤치마크 설정
- **Small 텐서**: (256, 256) - 10회 실행 평균
- **Medium 텐서**: (1024, 1024) - 10회 실행 평균
- **Large 텐서**: (4096, 4096) - 10회 실행 평균
- **최종 성능**: (Small 평균 + Medium 평균 + Large 평균) / 3
- **벤치마크 도구**: `triton.testing.do_bench` (warmup=25, rep=100)

## Phase 4: 최적 커널

### 선택된 커널
- **커널**: {selected_kernel}
- **최종 성능**: {final_performance} ms
- **Speedup**: {final_speedup}x

### 선정 이유
{selection_reason}

### 성능 분석
{performance_analysis}

### 개선 여지
{improvement_suggestions}
