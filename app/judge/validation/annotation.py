"""app/judge/validation/annotation.py — 사람 주장 정답지 표시지·안내서(계획 10-4, 설계 7.1 추출 재현율·정밀도).

표시자(인용 님)는 설계 문서를 열 수 없다 — 안내서는 기준 묶음(criteria/claims.json)에서 바로 만들어 원문 없이 읽힌다.
같은 정의를 두 곳에 따로 쓰지 않는다(추출 지시문도 같은 claims.json 에서 채운다).

표시지는 봉인 파일(요청·작업 결과)마다 두 개다.
- <tag>.sentences.csv: 추출 범위(AI·규칙 문장 조각)의 문장마다 번호. 읽기만 한다.
- <tag>.claims.csv: 표시자가 채운다 — 문장번호, 종류, 인용 구절, 메모.

상호작용
- 의존(import): app.judge.criteria·material·reading, app.narrative.schemas
- 사용처: scripts/judge_annotate.py
"""
from __future__ import annotations

import csv
import io
import re

from app.judge import criteria
from app.judge.material import parse_material
from app.judge.reading import Piece, build_pieces, for_module
from app.narrative.schemas import FactSheet, NarrativeJobResult

SENTENCE_COLUMNS = ("문장번호", "조각", "칸", "판4자리", "출처", "문장")
CLAIM_COLUMNS = ("문장번호", "종류", "인용 구절", "메모")
_SPLIT = re.compile(r"(?<=[.?!])\s+|\n+")


def pieces_from_sealed(record: dict) -> list[Piece]:
    """봉인 파일 하나(요청·작업 결과) → 조각 목록."""
    req = record["request"]
    view = record["view"]
    result = NarrativeJobResult.model_validate(view["result"])
    material, _ = parse_material(req.get("judge_material"))
    metrics = FactSheet.model_validate(req["fact_sheet"]).metrics
    return build_pieces(result, result.section_sources, material, view.get("benchmark_refs") or [], metrics)


def sentence_rows(pieces: list[Piece]) -> list[tuple]:
    rows = []
    for p in for_module(pieces, "extract"):
        for s in (x.strip() for x in _SPLIT.split(p.text)):
            if s:
                rows.append((len(rows) + 1, p.id, p.slot, p.place, p.source, s))
    return rows


def to_csv(columns, rows) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(columns)
    w.writerows(rows)
    return buf.getvalue()


def guide_markdown() -> str:
    rules = criteria.load_json("claims.json")
    kinds = "\n".join(f"| {k['id']} | {k['name']} | {k['what']} | {k['example']} |" for k in rules["kinds"])
    return f"""# 주장 표시 안내서 (기준 {criteria.version()})

이 안내서만 보고 표시할 수 있게 썼다. 성적서 서술의 **사실 주장**을 빠짐없이 찾아 적는 일이다 — 맞는지 틀린지는 보지 않는다.

## 할 일

1. `<성적서>.sentences.csv` 를 연다. 표시할 문장이 번호와 함께 있다(AI 가 쓴 칸과 규칙 문장 칸만).
2. 문장마다 아래 규칙대로 주장을 찾아 `<성적서>.claims.csv` 에 한 줄씩 적는다.
   - 문장번호: sentences.csv 의 번호
   - 종류: 아래 표의 id(number·name·comparison·degree·inference) 하나
   - 인용 구절: 그 주장이 드러나는 가장 짧은 구절을 문장에서 글자 그대로 옮긴다
   - 메모: 헷갈린 점(선택)
3. 주장이 없는 문장은 적지 않는다.

## 주장 종류

| id | 이름 | 무엇 | 예 |
|---|---|---|---|
{kinds}

## 나누는 규칙

{chr(10).join(f"- {r}" for r in rules["unit_rules"])}

## 주장이 아닌 것

{chr(10).join(f"- {r}" for r in rules["not_claims"])}

## 추론 표시 어미

{", ".join(rules["inference_markers"])} — 이런 어미로 끝나는 해석도 inference(추론) 주장으로 적는다(표시가 있는지는 따로 보지 않아도 된다).
"""
