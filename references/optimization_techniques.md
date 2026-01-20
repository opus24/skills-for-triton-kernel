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

## 조합 전략

### v1_baseline
- 최적화: 없음
- 특징: torch operation을 직접적으로 1:1 변환
- 용도: 성능 비교의 기준선

### v2_tiling
- 최적화: Tiling만 적용
- 특징: 
  - BLOCK_SIZE 파라미터로 타일링 구현
  - 데이터를 타일 단위로 분할하여 처리
  - 캐시 효율성 향상

### v3_coalesced
- 최적화: Memory Coalescing만 적용
- 특징:
  - 연속 메모리 접근 패턴 최적화
  - 벡터화된 로드/스토어 활용
  - 메모리 대역폭 최대화

### v4_optimized
- 최적화: Tiling + Memory Coalescing 조합
- 특징: 두 기법을 모두 적용하여 최대 성능 추구
