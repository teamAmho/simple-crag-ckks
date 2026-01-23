# 1) 코드 컨벤션 

## 1.1 스타일 규칙

* **Python**: 3.11+ (팀 통일)
* **Formatter**: `black` (line length 88)
* **Import Sort**: `isort` (black profile)
* **Lint**: `ruff`
* **Type Check**: `mypy` (필요한 모듈만 점진 도입)
* **Naming**: PEP8 준수

권장 실행 순서:

1. black
2. isort
3. ruff
4. mypy

---

## 1.2 폴더/모듈 구조 규칙

* `src/` 기반으로 모듈 관리
* 기능 단위로 분리 (ingestion / embeddings / retrieval / ckks / pipeline)
* `scripts/`는 **실행용**(CLI/데모)
* `src/`는 **라이브러리 코드**만 유지

예시:

```text
src/
  rag_ckks/
    retrieval/
    ckks/
    pipeline/
scripts/
tests/
```

---

## 1.3 네이밍 규칙

### 파일/폴더

* `snake_case.py`
* 폴더도 `snake_case/`

### 클래스

* `PascalCase`
* 예: `CkksEncryptor`, `VectorStore`

### 함수/변수

* `snake_case`
* 예: `build_index()`, `chunk_text`

### 상수

* `UPPER_SNAKE_CASE`
* 예: `DEFAULT_TOP_K = 5`

---

## 1.4 타입 힌트

* 핵심 인터페이스에만 최소 적용 (전부 강제 X)
* 공개 함수(모듈 외부에서 사용)에는 타입 붙이기

예시:

```python
def cosine_similarity(a: list[float], b: list[float]) -> float:
    ...
```

---

## 1.5 Docstring 규칙 
: 함수마다 주석 달기

* “팀원이 이해해야 하는 함수”만 작성
* 형식은 Google Style로 통일

예시:

```python
def encrypt_embedding(vec: list[float]) -> bytes:
    """Encrypt embedding vector using CKKS.

    Args:
        vec: Plain embedding vector.

    Returns:
        Serialized ciphertext bytes.
    """
```

---

## 1.6 예외 처리 규칙 
-> 현재는 에러 처리는 최소한만

* `except Exception:` 무조건 금지 (불가피하면 주석으로 이유 명시)
* 에러는 `raise ValueError(...)` 같이 의미 있는 타입으로

---

## 1.7 로깅 규칙 
-> 로깅도 마찬가지로 최소한, 다만 중간 과정을 확인할 수 있도록 처리

* `print()` 금지 (scripts는 예외적으로 허용)
* `logging` 사용, 라이브러리에서는 `logger = logging.getLogger(__name__)`

---

## 1.8 테스트 규칙

* `pytest` 사용
* “수학/암호 모듈”은 최소 단위 테스트 필수

  * cosine similarity
  * normalize
  * CKKS encrypt/decrypt roundtrip
  * retrieval top-k 안정성

---

# 2) Git 컨벤션

## 2.1 브랜치 전략

* `main`: 항상 동작해야 함 (배포/최종)
* `dev`: 통합 브랜치 (기능 병합)
* `feat/*`: 기능 개발
* `fix/*`: 버그 수정
* `chore/*`: 설정/문서/빌드
* `refactor/*`: 리팩토링

예시:

* `feat/ckks-encryptor`
* `feat/retriever-topk`
* `fix/cosine-zero-division`
* `chore/add-ruff-black`
* `refactor/vectorstore-interface`

---

## 2.2 커밋 메시지 규칙

형식:

```text
[TYPE] short summary
```

TYPE 목록:

* `[FEAT]` 기능 추가
* `[FIX]` 버그 수정
* `[REFACTOR]` 리팩토링 (기능 변화 없음)
* `[CHORE]` 설정/빌드/CI/의존성
* `[DOCS]` 문서
* `[TEST]` 테스트 추가/수정
* `[PERF]` 성능 개선
* `[STYLE]` 포맷/린트 (로직 변경 없음)

좋은 예시:

* `[FEAT] add CKKS encrypt/decrypt wrapper`
* `[FEAT] implement brute-force cosine retriever`
* `[FIX] handle zero norm in cosine similarity`
* `[REFACTOR] split rag pipeline into retrieval and generation`
* `[CHORE] setup black ruff isort pre-commit`
* `[TEST] add retrieval top-k deterministic test`

나쁜 예시:

* `[FEAT] update` (내용 불명확)
* `fix bug` (형식 불일치)
* `WIP` (dev 브랜치 외 금지)

---

## 2.3 PR 규칙

* PR 단위는 “기능 1개” 기준 (너무 크게 합치지 않기)
* 리뷰 승인 최소 1명
* PR 제목도 커밋과 동일한 prefix 사용

예시:

* `[FEAT] CKKS encrypted embedding storage`
* `[FIX] retrieval ranking bug`

---

## 2.4 PR 설명 템플릿

```text
## Summary
- 무엇을 했는지 2~3줄

## Changes
- 핵심 변경사항 리스트

## Test
- 어떤 테스트를 돌렸는지 (pytest 등)

## Notes
- 팀원이 알아야 할 주의점
```

---

## 2.5 머지 방식

* 반드시 인당 하나 이상의 branch 가지고 작업 
* `dev`로 merge: 각 팀별로 한 명이 관리(코드 리뷰) dev 머지 후 단톡으로 알려주세요.
* `main`으로 merge: `dev` 안정화 후 merge (조휘정이 관리, 다른 분은 main 머지 하지 말아주세요.)

---

## 2.6 코드리뷰 체크리스트

* 함수/클래스 이름이 역할을 설명하는가?
* CKKS 연산의 입력/출력 타입이 명확한가?
* 로그/예외 처리가 적절한가?
* 테스트가 최소 1개 이상 추가되었는가?

