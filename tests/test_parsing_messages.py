"""tests/test_parsing_messages.py — 업로드 파싱 오류 문구가 실제 원인을 말한다.

ISSUES.md D-13 · D-14 (2026-09-07 ★작은 판단 ② — "문구와 행 번호만 정정하고
**거절 규칙은 관대하게 유지**"). 흔한 엑셀 내보내기 파일이 거절되면 안 되므로
파싱 규칙 자체는 넓히지도 좁히지도 않는다.
"""
import json

import pytest

from app.core.parsing import parse_file_content


# ── D-13: 인코딩 폴백의 마지막 단계는 실패할 수 없다 ──────────────────────

def test_encoding_fallback_message_is_not_about_encoding_when_structure_is_broken():
    """`latin-1` 은 256바이트 전부를 디코딩하므로 UnicodeDecodeError 로 탈출하지 못한다.

    그 raise 에 도달하는 유일한 경로는 **구조가 깨진 CSV**(ParserError)인데, 문구는
    "파일 인코딩을 자동으로 감지할 수 없습니다"라고 말했다 — 원인과 무관한 안내를
    받은 사용자는 인코딩을 바꿔 가며 헛수고를 한다.
    """
    broken = b'a,b\n1,2\n3,4,5,6,7\n'
    with pytest.raises(ValueError) as exc:
        parse_file_content(broken, "broken.csv")

    message = str(exc.value)
    assert "인코딩" not in message, message
    assert "열" in message or "구조" in message, message


def test_cp949_file_still_parses():
    """한국어 Excel 내보내기(CP949)는 종전대로 읽힌다 — 거절 규칙을 좁히지 않는다."""
    columns, df = parse_file_content("이름,값\n가,1\n".encode("cp949"), "k.csv")
    assert columns == ["이름", "값"]
    assert len(df) == 1


def test_utf8_bom_file_still_parses():
    columns, _ = parse_file_content("﻿a,b\n1,2\n".encode("utf-8"), "bom.csv")
    assert columns == ["a", "b"]


def test_empty_csv_gets_a_korean_message_not_pandas_english():
    """공백만 있는 CSV 는 `EmptyDataError` 를 내는데 그것은 ParserError 의 자식이 아니라
    except 절을 그냥 통과했다 — pandas 영문 원문이 그대로 사용자에게 나갔다."""
    with pytest.raises(ValueError) as exc:
        parse_file_content(b"\n", "empty.csv")

    message = str(exc.value)
    assert "No columns to parse" not in message, message
    assert "비어" in message or "열" in message, message


# ── D-14: JSON 단일 키 언래핑이 형식 구분 없이 키 개수만 봤다 ─────────────

def test_single_column_json_dict_is_not_unwrapped_as_a_wrapper():
    """열이 하나뿐인 열 기반 dict 를 '래퍼'로 오인해 컬럼명을 잃었다.

    `{"y_true": [1, 0, 1]}` 는 컬럼 하나짜리 정상 데이터다. 종전 규칙("키가 1개이고
    값이 리스트면 언래핑")은 이것을 `[1, 0, 1]` 로 펼쳐 **컬럼명 없는 단일 열**을 만들었다.
    """
    columns, df = parse_file_content(json.dumps({"y_true": [1, 0, 1]}).encode(), "d.json")

    assert columns == ["y_true"]
    assert df["y_true"].tolist() == [1, 0, 1]


def test_records_wrapper_is_still_unwrapped():
    """진짜 래퍼(값이 dict 의 리스트)는 종전대로 벗긴다 — 관대함을 유지한다."""
    payload = {"samples": [{"y_true": 1, "y_pred": 0}, {"y_true": 0, "y_pred": 0}]}
    columns, df = parse_file_content(json.dumps(payload).encode(), "d.json")

    assert sorted(columns) == ["y_pred", "y_true"]
    assert len(df) == 2


def test_multi_column_dict_is_not_unwrapped():
    columns, df = parse_file_content(
        json.dumps({"y_true": [1, 0], "y_pred": [1, 1]}).encode(), "d.json"
    )
    assert sorted(columns) == ["y_pred", "y_true"]
    assert len(df) == 2


def test_records_array_still_parses():
    columns, df = parse_file_content(json.dumps([{"a": 1}, {"a": 2}]).encode(), "d.json")
    assert columns == ["a"]
    assert len(df) == 2


def test_invalid_json_gets_a_korean_message():
    with pytest.raises(ValueError) as exc:
        parse_file_content(b"{not json", "d.json")
    assert "JSON" in str(exc.value)
