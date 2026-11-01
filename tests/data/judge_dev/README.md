# 심판 개발용 판 4 서술 요청 (심판 라운드 J0)

심판을 만드는 동안(기준 고정 전) 시험·프롬프트·파일럿에 쓰는 입력이다. 검증용 성적서(판 3 발급본과 같은 입력 6건)는
검증 오염을 막으려고 저장소에 두지 않는다 — 심판 개발에는 이 세 파일과 합성 문장만 쓴다.

| 파일 | 유형 | 예측 CSV | 무엇이 다른가 |
|---|---|---|---|
| `dev_binary.request.json` | 이진 | `tests/data/binary/binary_test_data_200.csv`(Accuracy 0.505) | 평가 데이터 출처 두 줄(출처·학습 데이터와의 관계) 모두 제공, 위험 방향 FN, 외부 제출 |
| `dev_multiclass.request.json` | 다중(cat·dog·bird) | `tests/data/multiclass/multiclass_200.csv` | 평가 데이터 카드 끔, 실행 환경 있음 |
| `dev_multilabel.request.json` | 멀티레이블(4개) | `tests/data/multilabel/multilabel_200.csv` | 출처 한 줄만, 정답 라벨 카드 끔, 지표 선택 이유 있음 |

만든 법: 백엔드(키 없음)로 위 CSV 를 평가·검증한 응답과 가상의 의뢰 입력(스토어 상태)을 프론트에 넣고, 실제 `<Report />` 를
판 4 플래그로 그려 앱이 `POST /api/narrative/jobs` 로 보낸 본문을 `JSON.stringify(본문, null, 2) + "\n"` 로 떴다
(판 4 계약 예시 `tests/contract/fact_sheet_*_ed4.json` 과 같은 길 — 같은 도구로 뜬 강한 모델 3건이 계약 예시와 바이트까지 같았다).

요청 모양이 바뀌면 같은 도구로 다시 뜨고 `tests/test_judge_dev_fixtures.py` 의 sha256 을 이유와 함께 고친다.
