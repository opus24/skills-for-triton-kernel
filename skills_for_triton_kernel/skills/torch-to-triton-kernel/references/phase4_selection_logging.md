# Phase 4: 최적 커널 선정 및 로그 작성

## 목표
벤치마크 결과를 분석하여 최적 커널을 선정하고 전체 프로세스를 문서화

## 수행 작업

### 1. 결과 비교 테이블 생성

모든 커널과 torch baseline의 성능을 비교하는 테이블을 생성합니다:

```
| Kernel | Small (ms) | Medium (ms) | Large (ms) | Final (ms) | Speedup vs Torch |
|--------|------------|-------------|------------|------------|------------------|
| v1_baseline | ... | ... | ... | ... | ... |
| v2_tiling | ... | ... | ... | ... | ... |
| v3_coalesced | ... | ... | ... | ... | ... |
| v4_optimized | ... | ... | ... | ... | ... |
| torch (baseline) | ... | ... | ... | ... | 1.0x |
```

### 2. 최적 커널 선정

다음 기준으로 최적 커널을 선정합니다:

- **최종 성능이 가장 빠른 커널 선택**
  - Final (ms) 값이 가장 작은 커널
  - Correctness check를 통과한 커널만 고려
  
- **Speedup 기준**
  - Speedup이 1.0x 이상인 경우 성공으로 간주
  - Speedup이 1.0x 미만이어도 최선의 커널 선택

- **분석 및 설명**
  - 왜 해당 커널이 최적인지 설명
  - 최적화 기법의 효과 분석
  - 개선 여지 및 제안

### 3. 프로세스 로그 작성

`logs/{op_name}_log.md` 파일에 다음 내용을 기록합니다:

- **Phase 1**: 자료 조사 결과
  - Operation 정보
  - 수학적 정의
  - 입력/출력 사양
  - 계산 복잡도
  
- **Phase 2**: 커널 작성 과정 및 특징
  - 각 커널의 특징
  - 적용된 최적화 기법
  - BLOCK_SIZE 및 튜닝 파라미터
  
- **Phase 3**: Correctness check 결과
  - 각 커널의 통과 여부
  - 재시도 이력
  - 실패한 경우 에러 정보
  
- **Phase 3**: 벤치마크 결과 상세
  - 각 커널의 성능 측정 결과
  - Small/Medium/Large별 성능
  - torch 대비 speedup
  
- **Phase 4**: 최적 커널 선정 및 분석
  - 선택된 커널
  - 선정 이유
  - 성능 분석

## 로그 파일 구조

로그 파일은 다음 구조를 따릅니다:

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
- BLOCK_SIZE: N/A

### v2_tiling
- 특징: ...
- BLOCK_SIZE: 64

### v3_coalesced
- 특징: ...
- 최적화 기법: Memory Coalescing

### v4_optimized
- 특징: ...
- 최적화 기법: Tiling + Memory Coalescing
- BLOCK_SIZE: 64

## Phase 3: Correctness Check
- v1_baseline: ✅ 통과 (재시도 0회)
- v2_tiling: ✅ 통과 (재시도 1회)
- v3_coalesced: ✅ 통과 (재시도 0회)
- v4_optimized: ✅ 통과 (재시도 0회)

## Phase 3: 벤치마크 결과
| Kernel | Small | Medium | Large | Final | Speedup |
|--------|-------|--------|-------|-------|---------|
| v1_baseline | ... | ... | ... | ... | ...x |
| v2_tiling | ... | ... | ... | ... | ...x |
| v3_coalesced | ... | ... | ... | ... | ...x |
| v4_optimized | ... | ... | ... | ... | ...x |
| torch | ... | ... | ... | ... | 1.0x |

## Phase 4: 최적 커널
- 선택된 커널: v4_optimized
- 최종 성능: ... ms
- Speedup: ...x
- 분석: ...
```

## 체크리스트
- [ ] 결과 비교 테이블 생성 완료
- [ ] 최적 커널 선정 완료
- [ ] 선정 이유 및 분석 작성 완료
- [ ] Phase 1-4 모든 내용을 로그 파일에 기록 완료
- [ ] 로그 파일 구조가 템플릿과 일치하는지 확인 완료
