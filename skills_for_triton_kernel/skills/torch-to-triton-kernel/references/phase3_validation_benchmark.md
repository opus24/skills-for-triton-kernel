# Phase 3: Correctness Check 및 벤치마크

## 3-1. Correctness Check

### 목표
생성된 커널이 torch operation과 동일한 결과를 생성하는지 검증

### 검증 프로세스

각 커널에 대해 다음 프로세스를 수행:

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
                log_success(kernel, retry_count)
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

### 검증 기준
- `torch.allclose(triton_output, torch_output, atol=1e-5, rtol=1e-5)` 사용
- 다양한 shape, dtype 조합으로 테스트
- Edge case 포함 (빈 텐서, 단일 요소 등)
- 다양한 입력 크기 테스트

### 재생성 로직
Correctness check 실패 시:
1. 에러 정보 분석 (shape mismatch, dtype 문제, 수치 오차 등)
2. 문제 원인 파악
3. 커널 코드 수정 또는 재작성
4. 최대 3회 재시도
5. 3회 실패 시 해당 커널은 스킵하고 다음으로 진행

### 출력
- 각 커널의 correctness check 결과
- 재시도 이력
- 실패한 경우 상세한 에러 정보
- `logs/{op_name}_log.md`에 결과 기록

## 3-2. 벤치마크 실행

### 목표
4개 커널의 성능을 측정하여 최적 커널 선정

### 벤치마크 프로토콜

```python
TENSOR_SIZES = {
    "small": (256, 256),
    "medium": (1024, 1024),
    "large": (4096, 4096)
}
NUM_RUNS = 10

for kernel in [v1_baseline, v2_tiling, v3_coalesced, v4_optimized]:
    # Correctness check 통과한 커널만 벤치마크
    if not kernel.passed_correctness:
        continue
    
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

### 성능 측정 상세

- **Small 텐서**: (256, 256) - 10회 실행 후 평균
- **Medium 텐서**: (1024, 1024) - 10회 실행 후 평균
- **Large 텐서**: (4096, 4096) - 10회 실행 후 평균
- **최종 성능**: (Small 평균 + Medium 평균 + Large 평균) / 3
- **벤치마크 도구**: `triton.testing.do_bench` 사용
  - `warmup=25`: 25회 워밍업 실행
  - `rep=100`: 100회 측정 실행

### 비교 기준
- torch operation의 성능도 동일한 방식으로 측정
- Speedup = torch_time / triton_time 계산
- Speedup이 1.0x 이상인 경우 성공으로 간주

### 출력
- 각 커널의 벤치마크 결과 (Small/Medium/Large/최종)
- torch 대비 speedup 계산
- `logs/{op_name}_log.md`에 결과 기록

## 체크리스트
- [ ] 모든 커널에 대해 correctness check 실행
- [ ] 실패한 커널 재생성 (최대 3회)
- [ ] Correctness check 통과한 커널만 벤치마크 실행
- [ ] Small/Medium/Large 각 10회 실행 완료
- [ ] 최종 성능 계산 완료
- [ ] torch 대비 speedup 계산 완료
- [ ] 모든 결과를 로그 파일에 기록 완료
