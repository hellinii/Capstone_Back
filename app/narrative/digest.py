"""app/narrative/digest.py — 서술 글 지문(심판 라운드 J9, 계획 14-9b·Q1 다).

판 4 작업이 심판을 통과하면 그 작업의 서술 글 지문을 내부 기록(internal_json["judge"]["narrative_sha256"])에 남기고, 발급·정정 발급
때 요청 content(프론트 FinalReportData) 안의 서술 글로 같은 지문을 만들어 맞댄다 — "통과한 다른 작업 번호"나 서술을 바꾼 content 를
막는다(app.issuance.judge_gate).

무엇을 같게 보나(정규화) — 프론트가 작업 응답을 성적서 데이터로 옮기는 한 곳(Capstone_Front src/lib/report/narrativeJobApi.ts
mapNarrativeResult → src/hooks/useReportData.ts mergeNarrative)을 그대로 흉내 낸다. 프론트는 서술 글을 고치지 않고 칸에 옮겨
담기만 하며, 세로줄(D54)·옮겨 찍는 고정 문구는 그릴 때만 붙고 저장되지 않는다. 그래서:

1. 칸: interpretation(confusion_analysis·distribution_analysis·additional_observations) ↔ content.interpretation(confusionAnalysis·
   distributionAnalysis·additionalObservations), conclusion(benchmark·narrative·risks) ↔ content.conclusion 의 같은 이름(판정·점수·
   8.1 표 같은 코드 값은 빼고 글 세 칸만), recommendation_narrative(data_quality·model_ops) ↔ content.recommendationNarrative
   (dataQuality·modelOps), recommendations[] ↔ content.recommendations[](순서 그대로).
2. 글자는 바이트 그대로 — 공백·줄바꿈·유니코드 정규화를 하지 않는다(프론트도 하지 않음). 작업 쪽에서 글 칸이 없거나 null 이면 ""
   (프론트 `?? ""`).
3. 권고 행: priority 는 대문자로, HIGH·MEDIUM·LOW 밖이면 MEDIUM(프론트 normalizePriority). category·action·expected_impact 는 없으면
   ""(프론트 `String(x ?? "")`). problem_id 는 문자열이고 앞뒤 공백을 뗀 뒤 비어 있지 않을 때만 싣는다(프론트는 그때만 problemId 키를 둠).
4. content 쪽은 프론트가 저장한 값을 **그대로** 읽는다(다시 고치지 않는다) — 키가 없거나 문자열이 아니면 그 값 그대로 지문에 들어가
   작업 쪽과 달라진다(닫힌 쪽).

지문 = 위 모양을 키 정렬·한글 그대로(ensure_ascii=False)·공백 없는 JSON 으로 적은 UTF-8 의 sha256.

상호작용
- 의존(import): hashlib, json
- 사용처: app.narrative.jobs(판 4 통과 때 기록), app.issuance.judge_gate(발급·정정 때 대조), tests
"""
from __future__ import annotations

import hashlib
import json

PRIORITIES = ("HIGH", "MEDIUM", "LOW")


def _text(value) -> str:
    return "" if value is None else value


def _priority(value) -> str:
    s = ("" if value is None else str(value)).upper()
    return s if s in PRIORITIES else "MEDIUM"


def _problem_id(value):
    return value.strip() if isinstance(value, str) and value.strip() != "" else None


def from_result(result: dict) -> dict:
    """작업 결과(고객용 NarrativeJobResult 를 JSON 으로 적은 dict, snake_case) → 프론트가 저장할 서술 글 모양."""
    interp = result.get("interpretation") or {}
    concl = result.get("conclusion") or {}
    rec_n = result.get("recommendation_narrative") or {}
    rows = []
    for r in result.get("recommendations") or []:
        row = {
            "priority": _priority(r.get("priority")),
            "category": str(_text(r.get("category"))),
            "action": str(_text(r.get("action"))),
            "expectedImpact": str(_text(r.get("expected_impact"))),
        }
        pid = _problem_id(r.get("problem_id"))
        if pid is not None:
            row["problemId"] = pid
        rows.append(row)
    return {
        "interpretation": {
            "confusionAnalysis": _text(interp.get("confusion_analysis")),
            "distributionAnalysis": _text(interp.get("distribution_analysis")),
            "additionalObservations": _text(interp.get("additional_observations")),
        },
        "conclusion": {k: _text(concl.get(k)) for k in ("benchmark", "narrative", "risks")},
        "recommendationNarrative": {
            "dataQuality": _text(rec_n.get("data_quality")),
            "modelOps": _text(rec_n.get("model_ops")),
        },
        "recommendations": rows,
    }


_MISSING = {"__없음__": True}  # content 에 칸이 없을 때 — 어떤 글과도 같지 않다


def _pick(obj, key):
    return obj.get(key, _MISSING) if isinstance(obj, dict) else _MISSING


def from_content(content: dict) -> dict:
    """발급 요청 content(프론트 FinalReportData, camelCase) → 같은 모양. 값은 고치지 않고 그대로 옮긴다."""
    interp = _pick(content, "interpretation")
    concl = _pick(content, "conclusion")
    rec_n = _pick(content, "recommendationNarrative")
    recs = _pick(content, "recommendations")
    rows = []
    if isinstance(recs, list):
        for r in recs:
            if not isinstance(r, dict):
                rows.append(_MISSING)
                continue
            row = {k: _pick(r, k) for k in ("priority", "category", "action", "expectedImpact")}
            if "problemId" in r:
                row["problemId"] = r["problemId"]
            rows.append(row)
    else:
        rows = _MISSING
    return {
        "interpretation": {k: _pick(interp, k) for k in ("confusionAnalysis", "distributionAnalysis", "additionalObservations")},
        "conclusion": {k: _pick(concl, k) for k in ("benchmark", "narrative", "risks")},
        "recommendationNarrative": {k: _pick(rec_n, k) for k in ("dataQuality", "modelOps")},
        "recommendations": rows,
    }


def sha256(shape: dict) -> str:
    text = json.dumps(shape, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def of_result(result: dict) -> str:
    return sha256(from_result(result))


def of_content(content: dict) -> str:
    return sha256(from_content(content))
