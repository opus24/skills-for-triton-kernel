---
name: torch-to-triton-kernel
description: Converts PyTorch operations to optimized Triton kernels through a 4-phase pipeline (research, generation, validation, selection). Use when converting torch ops like softmax, layernorm, gelu to Triton kernels, when optimizing GPU computations, or when benchmarking kernel performance.
---

# Torch-to-Triton Kernel Agent Skills

## 개요

이 Skill은 사용자가 입력한 여러 torch operations를 Triton kernel로 변환하는 agent의 워크플로우를 정의합니다. 각 operation에 대해 4단계 파이프라인을 순차적으로 실행하여 최적화된 커널을 생성하고 평가합니다.

## 핵심 요구사항

- **다중 Ops 처리**: 사용자가 여러 ops를 한번에 입력 가능 (예: "softmax, layernorm, gelu"). 각 op은 Phase 1-4를 모두 거쳐야 함
- **벤치마크 방식**: Small/Medium/Large 텐서 각 10회 실행, 3가지 평균의 최종 평균을 최종 성능으로 사용
- **재생성 로직**: Correctness check 실패 시 해당 커널이 통과할 때까지 재생성 (최대 3회)

## 최적화 기법

2가지 최적화 기법을 조합하여 4개의 커널 변형을 생성합니다:
- **Tiling**: 데이터를 BLOCK_SIZE 단위 타일로 분할하여 캐시 효율성 극대화
- **Memory Coalescing**: 연속 메모리 접근 패턴으로 메모리 대역폭 활용률 극대화

**See**: `references/optimization_techniques.md` for detailed explanation of optimization techniques.

## 전체 워크플로우 (⚠️ MANDATORY)

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

입력된 torch operation에 대한 완전한 이해를 얻어 올바른 Triton kernel을 작성할 수 있도록 함. Operation 식별, 공식 문서 조사, 소스 코드 분석, 정리 및 문서화를 수행.

**See**: `references/phase1_research.md` for detailed task description and workflow.

## Phase 2: 4가지 Triton Kernel 작성

2가지 최적화 기법의 조합으로 총 4개의 커널 변형 생성 (v1_baseline, v2_tiling, v3_coalesced, v4_optimized). 각 커널은 `kernels/{op_name}/` 디렉토리에 생성되며 독립적으로 실행 가능해야 함.

**See**: `references/phase2_kernel_generation.md` for kernel structure, rules, and tuning parameters.

## Phase 3: Correctness Check 및 벤치마크

생성된 커널이 torch operation과 동일한 결과를 생성하는지 검증하고, 통과한 커널에 대해 성능 벤치마크를 실행. Correctness check 실패 시 최대 3회 재생성 시도.

**See**: `references/phase3_validation_benchmark.md` for validation protocol and benchmark procedures.

## Phase 4: 최적 커널 선정 및 로그 작성

벤치마크 결과를 분석하여 최적 커널을 선정하고 전체 프로세스를 문서화. 결과 비교 테이블 생성, 최적 커널 선정, 프로세스 로그 작성.

**See**: `references/phase4_selection_logging.md` for selection criteria and log structure.

## 다중 Ops 처리

사용자는 쉼표로 구분된 여러 ops를 입력할 수 있습니다 (예: "softmax, layernorm, gelu"). 각 op에 대해 **순차적으로** Phase 1-4를 모두 실행한 후 최종 요약 리포트를 출력합니다.

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
