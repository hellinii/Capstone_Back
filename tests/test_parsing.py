"""tests/test_parsing.py — app.core.parsing 순수 함수 단위 테스트.

PR-B 에서 analyzer.py 로부터 분리한 파일 파싱 로직의 포맷별 분기(CSV 인코딩 / JSON 3형태 /
미지원 확장자)를 개별 고정한다. golden(엔드포인트) 테스트를 보완하는 세밀한 안전망.
"""
import json

import pytest

from app.core.parsing import parse_file_content


def test_csv_utf8():
    content = "id,y_true,y_pred\nS1,1,0\nS2,0,0\n".encode("utf-8")
    cols, df = parse_file_content(content, "d.csv")
    assert cols == ["id", "y_true", "y_pred"]
    assert len(df) == 2


def test_csv_cp949_korean():
    """한국어 CP949(엑셀 기본) 인코딩 자동 감지."""
    content = "이름,점수\n가나,1\n다라,0\n".encode("cp949")
    cols, df = parse_file_content(content, "kor.csv")
    assert cols == ["이름", "점수"]
    assert len(df) == 2


def test_json_records_array():
    content = json.dumps([{"a": 1, "b": 2}, {"a": 3, "b": 4}]).encode("utf-8")
    cols, df = parse_file_content(content, "d.json")
    assert set(cols) == {"a", "b"}
    assert len(df) == 2


def test_json_single_key_wrapper_unwrapped():
    """{"samples": [...]} 형태는 배열로 언래핑된다."""
    content = json.dumps({"samples": [{"a": 1}, {"a": 2}, {"a": 3}]}).encode("utf-8")
    cols, df = parse_file_content(content, "d.json")
    assert cols == ["a"]
    assert len(df) == 3


def test_json_column_based_dict():
    content = json.dumps({"a": [1, 2], "b": [3, 4]}).encode("utf-8")
    cols, df = parse_file_content(content, "d.json")
    assert set(cols) == {"a", "b"}
    assert len(df) == 2


def test_unsupported_extension_raises():
    with pytest.raises(ValueError):
        parse_file_content(b"x", "d.txt")
