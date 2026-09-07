"""app/core/parsing.py — 업로드 파일(CSV/JSON) → DataFrame 파싱

확장자에 따라 CSV(인코딩 자동 감지) 또는 JSON(여러 형태 정규화)을 pandas DataFrame 으로
읽어 컬럼 목록과 함께 반환한다. LLM 의존이 없는 순수 파싱 로직이라 포맷별로 단위 테스트가 가능하다.

상호작용
- 의존(import): pandas (표준 io/json)
- 사용처: app.analysis.router / app.analysis.validation_router / app.evaluation.router
  (업로드 파일을 df 로 변환), scripts.llm_smoke_analyze
"""
import io
import json

import pandas as pd


def _read_csv_any_encoding(file_content: bytes) -> pd.DataFrame:
    """CSV 바이트를 인코딩 자동 감지로 읽는다: UTF-8 → CP949(한국어 Excel) → latin-1 순.

    **마지막 단계 `latin-1` 은 인코딩 때문에 실패할 수 없다** — 256바이트 전부에 대응하는
    문자가 있어 `UnicodeDecodeError` 가 나지 않는다. 그래서 루프를 빠져나오는 유일한
    경로는 `ParserError`(구조가 깨진 CSV)인데, 종전 문구는 "파일 인코딩을 자동으로
    감지할 수 없습니다"라고 말했다(ISSUES.md D-13). 원인과 무관한 안내를 받은 사용자는
    인코딩을 바꿔 가며 헛수고를 한다.

    `EmptyDataError` 는 `ParserError` 의 자식이 **아니라서** except 절을 그냥 통과했고,
    pandas 영문 원문("No columns to parse from file")이 그대로 사용자에게 나갔다.

    거절 규칙 자체는 넓히지도 좁히지도 않는다 — 흔한 엑셀 내보내기 파일이 거절되면
    안 된다(★작은 판단 ②).
    """
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "cp949", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(file_content), encoding=encoding)
        except pd.errors.EmptyDataError:
            raise ValueError(
                "CSV 파일에서 읽을 수 있는 열이 없습니다. 첫 줄에 컬럼명이 있는지 확인해 주세요."
            )
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            last_error = e
            continue
    raise ValueError(
        "CSV 구조를 해석할 수 없습니다. 줄마다 열 개수가 다르거나 따옴표가 짝이 맞지 "
        f"않는지 확인해 주세요: {last_error}"
    )


def _looks_like_records(value) -> bool:
    """records 배열(=dict 의 리스트)인가. 스칼라 리스트는 '열 하나'이지 래퍼가 아니다."""
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(item, dict) for item in value)
    )


def _json_to_df(file_content: bytes) -> pd.DataFrame:
    """JSON 바이트를 DataFrame 으로. records 배열 / 열 기반 dict / 단일 키 래핑을 정규화한다."""
    try:
        raw = json.loads(file_content.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 형식 오류: 파일을 JSON 으로 읽을 수 없습니다({e.msg}).")
    except UnicodeDecodeError:
        raise ValueError("JSON 형식 오류: 파일이 UTF-8 로 인코딩되어 있어야 합니다.")

    if isinstance(raw, dict):
        values = list(raw.values())
        # 래퍼 판정은 **키 개수가 아니라 값의 형태**로 한다(ISSUES.md D-14).
        # `{"samples": [{...}, ...]}` 는 래퍼지만 `{"y_true": [1, 0, 1]}` 은 컬럼이
        # 하나뿐인 정상 데이터다. 종전 규칙("키 1개 + 값이 리스트")은 후자를 펼쳐
        # **컬럼명 없는 단일 열**로 만들었다 — 매핑 화면에서 그 컬럼이 사라진다.
        if len(values) == 1 and _looks_like_records(values[0]):
            raw = values[0]

    if isinstance(raw, list):
        return pd.DataFrame(raw)
    if isinstance(raw, dict):
        return pd.DataFrame(raw)
    raise ValueError(
        "JSON 형식 오류: records 배열([{...}]) 또는 열 기반 dict({col: [...]}) 형태여야 합니다."
    )


def parse_file_content(file_content: bytes, filename: str) -> tuple[list[str], pd.DataFrame]:
    """
    CSV 또는 JSON 파일을 파싱해 컬럼명 목록과 전체 DataFrame을 반환합니다.

    지원 JSON 구조:
      1. records 배열:  [{col: val, ...}, ...]
      2. 열 기반 dict:  {col: [val, ...], ...}
      3. 단일 키 래핑:  {"samples": [{...}, ...]}  ← 자동 언래핑
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "csv":
        df = _read_csv_any_encoding(file_content)
    elif ext == "json":
        df = _json_to_df(file_content)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: .{ext}  (CSV 또는 JSON만 허용)")

    # 전체 DataFrame 반환 (메타데이터 추출용)
    return df.columns.tolist(), df
