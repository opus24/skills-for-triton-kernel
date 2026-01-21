# skills-for-triton-kernel

PyTorch operations를 Triton kernels로 변환하는 4단계 파이프라인(연구, 생성, 검증, 선정)을 제공하는 Claude 스킬입니다.

## 설치

```bash
uv run skills-for-triton-kernel
```

인자 없이 실행하면 `~/.claude/skills/`에 `torch-to-triton-kernel` 스킬이 설치됩니다.

cursor에서 설치하려면 사용하려는 디렉토리의 홈에 복사해주세요.

```bash
cp -r ~/.claude/skills/torch-to-triton-kernel .
```

## 사용법

```bash
# 기본: ~/.claude/skills에 설치
uv run skills-for-triton-kernel

# install 서브커맨드 (--target, --force, --dry-run)
uv run skills-for-triton-kernel install
uv run skills-for-triton-kernel install --target /path/to/.claude/skills --force
uv run skills-for-triton-kernel install --dry-run

# 설치된 스킬 목록
uv run skills-for-triton-kernel list
```

## 스킬 구조 (설치 후)

```
~/.claude/skills/torch-to-triton-kernel/
├── SKILL.md
├── examples.md
└── references/
    ├── phase1_research.md
    ├── phase2_kernel_generation.md
    ├── phase3_validation_benchmark.md
    ├── phase4_selection_logging.md
    └── optimization_techniques.md
```

## 프로젝트 구조

- `skills_for_triton_kernel/` — 스킬 패키지 및 CLI
- `kernels/` — Triton 커널 예시 (add, softmax, mha)
- `tests/` — correctness, benchmark 테스트
- `references/` — 개발용 상세 참고 문서 (스킬의 `references/`와 동일)
- `assets/` — 로그 템플릿 및 예시

## 요구사항

- Python >= 3.10
- torch >= 2.0.0, triton >= 2.1.0
