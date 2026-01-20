# Phase 1: 자료 조사 및 분석

## 목표
입력된 torch operation에 대한 완전한 이해를 얻어 올바른 Triton kernel을 작성할 수 있도록 함

## 수행 작업

### 1. Operation 식별 및 파싱
- 사용자 입력에서 operation 이름 추출 (예: "softmax", "torch.nn.functional.softmax", "F.softmax")
- torch 모듈에서 해당 operation의 정확한 위치 확인
- 다양한 입력 형식 처리 (단축형, 전체 경로, 별칭 등)

### 2. 공식 문서 조사
- PyTorch 공식 문서에서 operation의 API, 파라미터, 동작 방식 확인
- 수학적 정의 및 알고리즘 설명 파악
- 공식 문서의 예시 코드 및 사용법 검토
- 버전별 차이점 확인 (필요시)

### 3. 소스 코드 분석
- torch의 C++/Python 구현 코드 검토 (가능한 경우)
- 입력/출력 텐서의 shape, dtype, 메모리 레이아웃 파악
- 연산의 계산 복잡도 및 메모리 접근 패턴 분석
- 내부 구현 로직 이해

### 4. 정리 및 문서화
다음 정보를 정리하여 `logs/{op_name}_log.md`에 Phase 1 섹션으로 기록:

- **Operation 이름 및 정확한 import 경로**
  - 예: `torch.nn.functional.softmax`
  
- **입력 파라미터**
  - shape 요구사항
  - dtype 요구사항
  - 기타 옵션 (dim, keepdim 등)
  
- **출력 shape 및 dtype**
  - 입력과의 관계
  - 변환 규칙
  
- **수학적 정의 (공식)**
  - 연산의 수학적 표현
  - 수식 및 알고리즘
  
- **계산 복잡도**
  - 시간 복잡도: O(?)
  - 공간 복잡도: O(?)
  
- **메모리 접근 패턴**
  - 순차 접근 vs 랜덤 접근
  - 메모리 지역성 특성
  - 병렬화 가능성

## 출력
- `logs/{op_name}_log.md`에 Phase 1 섹션 작성
- 분석 결과를 구조화된 형태로 기록
- 다음 Phase에서 참조할 수 있도록 명확하게 문서화

## 체크리스트
- [ ] Operation 이름 정확히 식별됨
- [ ] 공식 문서에서 API 및 파라미터 확인 완료
- [ ] 수학적 정의 파악 완료
- [ ] 입력/출력 shape 및 dtype 명확히 정리됨
- [ ] 계산 복잡도 분석 완료
- [ ] 메모리 접근 패턴 분석 완료
- [ ] 로그 파일에 Phase 1 섹션 작성 완료
