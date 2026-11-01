"""app/judge/material.py — 판 4 서술 요청의 심판 읽기 재료(`judge_material`)를 읽는 한 곳(계획 10-1).

프론트가 판 4 성적서에 찍는 글 가운데 서술 생성 결과에 없는 것 — 읽는 범위(설계 4절) 안의 고정 문구와 표 — 를 찍힌 글 그대로
실어 보낸다. AI 칸·규칙 문장 글은 백엔드가 생성 결과로 가지고 있어 싣지 않는다. 4.6 참조 범위 표는 백엔드가 만든다(정4).

형식이 틀리거나 상한을 넘으면 재료 전체를 버린다(서술은 그대로, 심판은 보류 — 재료 없음). 요청 검증(422)이나 서술 줄이기
(input_too_large)로 가지 않는다. 여기서 정한 모양은 두 저장소 계약(tests/contract/judge_material_ed4.json)과 같다.

조각 하나 = 찍힌 글 한 덩어리(고정 문구 한 문장 묶음 또는 표 하나).
- slot: 판 3 칸 이름(7.1·7.2·7.3·8.1·8.2·8.3·9) 또는 "8"(8절 머리 — 판 4 의 4.2 종합 판정). 판 4 로 옮겨 간 것도 판 3 에서
  놓였던 칸 이름을 쓴다(설계 4절 읽는 범위, 정5).
- place: 판 4 에서 찍힌 자리(예 "4.4", "5.2") — 부록 체크리스트가 실제 인쇄 자리를 쓰는 데 쓴다.
- kind: fixed(고정 문구) | table(표 — header 와 rows) | recommend_rows(6절 권고표의 프론트 몫 — 우선순위·대응 문제 칸만,
  row_keys 가 문제 번호. 권고·기대 효과 칸은 백엔드가 서술 결과로 채운다).
- when: always | slot_empty — slot_empty 는 그 칸 서술이 빈 글일 때만 찍히는 고정 문구(7.3 의 고정 15 '해당 없음', 8.1 의 B18).
- numbers: 글에 찍힌 숫자마다 {text, field}. field 는 숫자가 온 데이터 칸 이름(모르면 null) — 일관성 값 묶기에 쓴다.

상호작용
- 의존(import): pydantic, json
- 사용처: app.judge.reading
"""
from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

MATERIAL_VERSION = "judge-material-1"
MAX_MATERIAL_CHARS = 300_000  # 공백 없는 JSON 글자 수. 가장 큰 판 4 픽스처의 몇 배 — 본문 상한(2MB) 안
MAX_SEGMENTS = 120
MAX_TEXT_CHARS = 4_000
MAX_ROWS = 300
MAX_CELLS = 16
MAX_NUMBERS = 400

Slot = Literal["7.1", "7.2", "7.3", "8.1", "8.2", "8.3", "9", "8"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MaterialNumber(_Strict):
    text: str = Field(min_length=1, max_length=32)
    field: str | None = Field(default=None, max_length=96)


class MaterialSegment(_Strict):
    id: str = Field(min_length=1, max_length=64)
    slot: Slot
    place: str = Field(min_length=1, max_length=16)
    kind: Literal["fixed", "table", "recommend_rows"]
    when: Literal["always", "slot_empty"] = "always"
    text: str = Field(default="", max_length=MAX_TEXT_CHARS)
    header: list[str] = Field(default=[], max_length=MAX_CELLS)
    rows: list[list[str]] = Field(default=[], max_length=MAX_ROWS)
    row_keys: list[str] = Field(default=[], max_length=MAX_ROWS)
    numbers: list[MaterialNumber] = Field(default=[], max_length=MAX_NUMBERS)

    @field_validator("rows")
    @classmethod
    def _row_width(cls, v: list[list[str]]) -> list[list[str]]:
        if any(len(r) > MAX_CELLS for r in v):
            raise ValueError("표 한 줄의 칸이 너무 많다")
        return v

    @model_validator(mode="after")
    def _shape(self) -> "MaterialSegment":
        if self.kind == "fixed" and (self.header or self.rows or self.row_keys):
            raise ValueError("고정 문구 조각에 표 칸이 있다")
        if self.kind != "fixed" and not self.header:
            raise ValueError("표 조각에 머리 줄이 없다")
        if self.kind == "recommend_rows" and len(self.row_keys) != len(self.rows):
            raise ValueError("권고표 줄 수와 문제 번호 수가 다르다")
        if self.kind == "table" and self.row_keys:
            raise ValueError("일반 표에는 row_keys 를 두지 않는다")
        return self


class JudgeMaterial(_Strict):
    version: Literal["judge-material-1"]
    material_check_version: str | None = Field(default=None, max_length=32)
    segments: list[MaterialSegment] = Field(max_length=MAX_SEGMENTS)

    @field_validator("segments")
    @classmethod
    def _ids_unique(cls, v: list[MaterialSegment]) -> list[MaterialSegment]:
        ids = [s.id for s in v]
        if len(ids) != len(set(ids)):
            raise ValueError("조각 id 가 겹친다")
        return v


def parse_material(raw: Any) -> tuple[JudgeMaterial | None, str | None]:
    """(재료, 문제). 문제는 None(정상) · missing · too_large · invalid. 문제가 있으면 재료는 None."""
    if raw is None:
        return None, "missing"
    try:
        size = len(json.dumps(raw, ensure_ascii=False, separators=(",", ":")))
    except (TypeError, ValueError):
        return None, "invalid"
    if size > MAX_MATERIAL_CHARS:
        return None, "too_large"
    try:
        return JudgeMaterial.model_validate(raw), None
    except ValidationError:
        return None, "invalid"
