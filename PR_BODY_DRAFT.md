### 요약
이번 PR은 모놀리식 Lucia 서버의 테스트 안정성, 모듈화, 그리고 개발용 실행 편의성을 개선합니다.

주요 변경사항
- `ai_providers.py`: 테스트용 per-call mock과 mock_error, 강제 오류('will error') 처리를 우선 적용하여 테스트 결정성을 보장하고, 전역 `AI_USE_MOCK`는 실제 프로바이더가 구성된 경우에만 적용되도록 수정했습니다.
- `tests/test_ai_providers_custom.py`: provider-order 관련 테스트를 보정했습니다.
- `lucia_quantum_fusion.py`, `lucia_app.py`, 실행 스크립트 등 개발 편의용 파일을 추가했습니다. (개별 파일 목록은 커밋 참조)

테스트
- 로컬 pytest 전체 실행에서 통과하도록 조정했습니다. (사용자가 로컬에서 확인)
- 단위 테스트: `tests/test_ai_providers_custom.py` 포함

배경 및 동기
- CI/로컬 환경에서 네트워크 호출을 피하면서도, 테스트가 프로바이더 실패/롤백 경로를 검증할 수 있어야 합니다.
- 기존에는 전역 mock 설정이 stub-기대 테스트를 덮어써서 일부 테스트가 의도치 않게 통과하지 못했습니다.

주의사항
- `tests/conftest.py`가 기본적으로 `AI_USE_MOCK=true`를 설정합니다. 실환경에서 실제 프로바이더를 사용할 경우 해당 env를 변경하세요.

권장하는 병합/후속작업
- CI에 추가 로깅(artifacts) 업로드를 추가하면 실패 원인 분석이 쉬워집니다.
- `ai_quota`와 엔드포인트의 로그 레벨/포맷을 structured JSON로 통일하면 자동화 디버깅이 간편해집니다.

---
