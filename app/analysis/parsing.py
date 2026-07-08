"""app/analysis/parsing.py — 업로드 파일(CSV/JSON) → DataFrame 파싱

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
    """CSV 바이트를 인코딩 자동 감지로 읽는다: UTF-8 → CP949(한국어 Excel) → latin-1 순."""
    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "cp949", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(file_content), encoding=encoding)
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            last_error = e
            continue
    raise ValueError(f"파일 인코딩을 자동으로 감지할 수 없습니다: {last_error}")


def _json_to_df(file_content: bytes) -> pd.DataFrame:
    """JSON 바이트를 DataFrame 으로. records 배열 / 열 기반 dict / 단일 키 래핑을 정규화한다."""
    raw = json.loads(file_content.decode("utf-8"))

    if isinstance(raw, dict):
        values = list(raw.values())
        if len(values) == 1 and isinstance(values[0], list):
            raw = values[0]  # {"samples": [{...}, ...]} → [{...}, ...] 언래핑

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
