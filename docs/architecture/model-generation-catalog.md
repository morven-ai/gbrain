# 모델 세대 카탈로그

## 현재 카탈로그 (2026-09-23 확인)

가격 카탈로그와 provider recipe에 GPT-6 세 모델 및 Claude Opus 5.5를 추가했다. 구 모델 행과 recipe 항목은 사용 기록·기존 설정 호환을 위해 유지한다. OpenAI 자동 선택은 GPT-5.6의 `sol/terra/luna`와 GPT-6의 `astra/sol/luna` 계층을 세대별로 분류한다. 일반 선택 기본값과 기본 평가 패널은 새 모델로 갱신했다.

Opus 5.5 가격은 input/output `$4/$20`, cache read `$0.20`, 5분 cache write `$5`, 1시간 write `$8` per 1M tokens다. 카탈로그는 두 write rate를 보존한다. 사용량 집계가 cache-write TTL을 구분하지 않아 비용 추정에는 기본 5분 요금만 적용된다.

Anthropic expansion은 `generateObject`를 schema tool로 보내며 현재 AI SDK는 이를 `tool_choice: any`로 변환한다. Opus 5.5가 이 값을 400으로 거부한다는 호환성 제약은 모델별 분기로 처리하지 않았다. 기본 경로는 `thinking` 비활성화나 `budget_tokens`를 보내지 않는다.

## 검증

가격·recipe·기본 모델 관련 테스트는 437개 통과했다. 변경 TypeScript 파일을 진입점으로 한 검사에서 HEAD와 작업 트리 모두 `src/core/import-file.ts`의 같은 4개 외부 타입 진단(heic-decode, ImageData, AVIF WASM)을 냈고, 변경 파일 진단은 없었다. 저장소 전체 incremental typecheck는 이 실행 환경에서 종료되지 않았다.

## 롤백

DB 변경은 없다. 모델 기본값·카탈로그는 코드 롤백으로 되돌릴 수 있고 과거 가격 행은 유지된다. 새 코드가 OpenAI 계정 모델 캐시에 GPT-6 tier를 기록한 뒤 롤백한다면, OpenAI model discovery를 다시 갱신하거나 해당 캐시를 초기화한 후 사용한다. 새 기본값이 반영되려면 새 빌드/설치가 필요하며 상주 프로세스는 재시작해야 한다.
