"""app/judge/validation/evaluation.py — 판정자 평가 실험 집계(사전 등록 v1 §6). 순수 함수(파일·네트워크 없음).

사전 등록(평가/사전등록/v1/사전등록_v1.md)의 집계 규칙을 글자 그대로 옮긴다. 규칙에 없는 집계는 넣지 않는다 — 넣으면 이름에
"사후 분석"을 붙인다. 모호한 곳은 가장 문자 그대로의 읽기를 택하고 그 자리에 적는다(해석 메모).

- 6.1 문장 대조: 문서의 서술 조각(출처 ai·template)을 annotation.sentence_rows(정규식 `(?<=[.?!])\\s+|\\n+`)로 나눠 문장 번호를 붙인다
  (DocIndex). 판정자 표시의 인용은 stats.touches 로 문장에 붙인다 — 같은 조각의 문장을 먼저 보고, 거기 없거나 조각을 모르면
  문서 전체에서 찾는다(locate). 해석 메모: 조각이 서술 밖(고정 문구·표)으로 알려진 인용은 서술 문장에 붙이지 않는다(빈 목록).
- 주입 문장: 바탕 칸 글 ↔ 바뀐 칸 글을 같은 정규식으로 나눠 새로 생기거나 바뀐 문장(injected_sentences). 순서만 바뀌면 자리가
  바뀐 문장 모두.
- 6.2 탐지(detect): 게이트 수준·축 수준(C01 은 두 축 따로)·위치 불일치 표시. 실패 회차는 "실패"(미탐지로 세지 않음).
  두 회차 묶기(two_run_class), Wilson 95% 구간(app.evaluation.metrics.intervals.wilson_interval).
- 6.3 미탐지 원인(miss_cause): 추출 누락 → 필드 오지정 → 대상 오답 → 규칙 허용 → 모순 미검출 순서로 가장 앞선 것.
- 6.4 과잉 차단 판정표(sheet_entries·build_sheet·sheet_leaks)와 셈(tally_overblocking).
- 6.5 재현성(reproducibility), 6.6 불변성(invariance), 6.7 독립성(independence), 6.8 주입 문장 추출률(extraction_hit),
  6.9 숫자 토큰 재현율(numeric_tokens·numeric_recall).
- 5 베이스라인 1·2 탐지(baseline1_detection·baseline2_detection) — 1차 점검 실행 자체는 서술 쪽 규칙(app.narrative.checks)을
  쓰므로 scripts/judge_eval.py 가 하고(심판 패키지는 서술 규칙을 import 하지 않는다), 여기는 찾은 위치로 탐지만 센다.

LLM 을 부르지 않는다. 앱 실행 경로에서 import 되지 않는다(tests/test_judge_import_rules.py).

상호작용
- 의존(import): app.judge.validation 의 annotation·stats·variants·sheets, app.judge.reading(Piece·칸 표), app.judge.sources(ALIASES),
  app.evaluation.metrics.intervals(wilson_interval)
- 사용처: scripts/judge_eval.py, 시험 tests/test_judge_evaluation.py·tests/test_judge_eval_tool.py
"""
from __future__ import annotations

import difflib
import random
import re
from dataclasses import dataclass, field

from app.evaluation.metrics.intervals import wilson_interval
from app.judge.reading import NARRATIVE_PATHS, RECOMMEND_PATHS, Piece
from app.judge.sources import ALIASES
from app.judge.validation import annotation as A
from app.judge.validation import variants as V
from app.judge.validation.sheets import slot_of_path
from app.judge.validation.stats import AXES, axis_scores, touches

# ── 사전 등록 v1 에서 옮긴 값 ─────────────────────────────────────────────

PREREG_FINGERPRINT = "980e9bccd735aa3ce2713cc36426344d93bc606e432b03a3b9d179ecee93a4b9"  # §0 기준 1.25 지문
PREREG_JUDGE_MODEL, PREREG_JUDGE_EFFORT = "gpt-6.1-sol", "medium"  # §0
PREREG_EXTRACT_MODEL, PREREG_EXTRACT_EFFORT = "gpt-6-luna", "low"  # §0
EVAL_REPORTS = ("pima_strong", "segment_weak", "scene_strong")  # §2
DEV_REPORTS = ("bc_strong", "digits_weak", "yeast_strong")  # §2 대응 개발 결과서(판정표에 새면 안 되는 이름)
REPORT_TAGS = EVAL_REPORTS + DEV_REPORTS
BASE = "base"  # 원본 문서의 변형 칸 값(장부 variant)
ROUND_SEEDS = {1: 20261105, 2: 20261106}  # §7.2
BUDGET_USD, BUDGET_RUNS = 30.0, 110  # §7.7 이 실험 장부 상한
EST_USD_PER_RUN = 0.30  # 실행기 시작 전 예상(회당) — 계획의 평균 $0.27 위

EVAL_ID = re.compile(r"(F|C|V)\d{2}")  # 평가 결함(F·C)·변형(V) 폴더 이름 — 검증 단계 D·S 와 따로
FAITH, CONS = "faithfulness", "consistency"
GATE_AXES = (FAITH, CONS)
BLOCKING_VERDICTS = ("violation", "inference_violation")


@dataclass(frozen=True)
class PlannedDefect:
    report: str
    axis: str
    type: str
    expected_axes: tuple[str, ...]
    allowed_secondary_axes: tuple[str, ...]


def _d(report, axis, kind, expected, allowed=()):
    return PlannedDefect(report, axis, kind, tuple(expected), tuple(allowed))


# §3 결함 18개(차단 축만). 결과서는 §3.5 대로 옮길 수 있다(plant 가 기록) — 축·기대 축·허용 부차 축은 바꿀 수 없다.
PREREG_DEFECTS: dict[str, PlannedDefect] = {
    "F01": _d("pima_strong", FAITH, "숫자", [FAITH], [CONS]),
    "F02": _d("segment_weak", FAITH, "숫자", [FAITH], [CONS]),
    "F03": _d("segment_weak", FAITH, "이름", [FAITH]),
    "F04": _d("scene_strong", FAITH, "이름", [FAITH]),
    "F05": _d("pima_strong", FAITH, "비교·방향", [FAITH], [CONS]),
    "F06": _d("scene_strong", FAITH, "비교·방향(비율)", [FAITH]),
    "F07": _d("segment_weak", FAITH, "정도 표현", [FAITH], [CONS]),
    "F08": _d("pima_strong", FAITH, "단정형 추론", [FAITH]),
    "F09": _d("scene_strong", FAITH, "단정형 추론", [FAITH]),
    "C01": _d("pima_strong", CONS, "값 불일치", [FAITH, CONS]),
    "C02": _d("segment_weak", CONS, "판정↔권고", [CONS], [FAITH]),
    "C03": _d("scene_strong", CONS, "판정↔권고", [CONS], [FAITH]),
    "C04": _d("pima_strong", CONS, "약점↔권고", [CONS], [FAITH]),
    "C05": _d("segment_weak", CONS, "약점↔권고", [CONS], [FAITH]),
    "C06": _d("scene_strong", CONS, "약점↔권고", [CONS], [FAITH]),
    "C07": _d("pima_strong", CONS, "분석↔소견", [CONS], [FAITH]),
    "C08": _d("segment_weak", CONS, "분석↔소견", [CONS], [FAITH]),
    "C09": _d("scene_strong", CONS, "분석↔소견", [CONS], [FAITH]),
}
DEFECT_IDS = tuple(PREREG_DEFECTS)
VARIANT_IDS = tuple(f"V{i:02d}" for i in range(1, 13))  # §4 변형 12개
VARIANT_TYPES = ("연결 어미 교체", "숫자 표기 변환", "문장 순서 교체", "어휘 치환")  # §4

# §6.3 "결함과 관련된 종류" — C02~C09 는 일관성 경로라 없음(None: 6.8 은 어떤 종류든 참고로만)
RELATED_KINDS: dict[str, tuple[str, ...] | None] = {
    **{i: ("number",) for i in ("F01", "F02", "C01")},
    **{i: ("name",) for i in ("F03", "F04")},
    **{i: ("comparison", "number") for i in ("F05", "F06")},
    "F07": ("degree",),
    **{i: ("inference",) for i in ("F08", "F09")},
    **{f"C{i:02d}": None for i in range(2, 10)},
}
EXTRACTION_IDS = ("F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09", "C01")  # §6.8 (20회)

# §6.6 허용 폭 — 비율 점수 0.05, 건수 점수 1건
TOLERANCE = {"faithfulness": 0.05, "coverage": 0.05, "consistency": 1.0, "readability": 1.0}
FLOAT_EPS = 1e-9

# 분류 이름(보고서 글자)
DETECTED, MISSED, FAILED, NOT_RUN = "탐지", "미탐지", "실패", "미실행"
BOTH, ONE, NEITHER = "두 회차 모두 탐지", "한 회차만 탐지", "두 회차 모두 미탐지"
SOURCE_ORIGINAL, SOURCE_DEFECT, SOURCE_VARIANT, SOURCE_B2 = "원본", "결함본", "변형", "베이스라인2"


def doc_id(report: str, variant: str) -> str:
    """문서 이름 — 원본은 결과서 태그, 결함본·변형본은 그 ID(F01·V03 …)."""
    return report if variant == BASE else variant


def source_of(variant: str) -> str:
    if variant == BASE:
        return SOURCE_ORIGINAL
    return SOURCE_VARIANT if variant.startswith("V") else SOURCE_DEFECT


_SPACE = re.compile(r"\s+")


def _norm(text) -> str:
    return _SPACE.sub(" ", str(text or "")).strip()


# ── 6.1 문장 대조 ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Sentence:
    id: int
    piece_id: str
    slot: str
    place: str
    source: str
    text: str


def _as_piece(p) -> Piece:
    if isinstance(p, Piece):
        return p
    return Piece(p["id"], p["slot"], p.get("place", ""), p["source"], p.get("text", ""))


@dataclass
class DocIndex:
    """문서 하나의 서술 문장 목록(문장 번호 1부터)과 조각 출처표(서술 밖 조각을 알아보려고)."""

    sentences: list[Sentence]
    piece_sources: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_pieces(cls, pieces) -> "DocIndex":
        ps = [_as_piece(p) for p in pieces]
        return cls([Sentence(*row) for row in A.sentence_rows(ps)], {p.id: p.source for p in ps})

    def get(self, sid: int) -> Sentence:
        return self.sentences[sid - 1]

    def locate(self, quote: str, piece_id: str = "") -> list[int]:
        """인용 → 문장 번호들(stats.touches). 같은 조각 먼저, 거기 없거나 조각을 모르면 문서 전체. 서술 밖 조각의 인용은 []."""
        if not _norm(quote):
            return []
        if piece_id:
            same = [s.id for s in self.sentences if s.piece_id == piece_id and touches(quote, s.text)]
            if same:
                return same
            if piece_id in self.piece_sources and self.piece_sources[piece_id] not in ("ai", "template"):
                return []  # 고정 문구·표 인용 — 서술 문장이 아니다(해석 메모)
        return [s.id for s in self.sentences if touches(quote, s.text)]

    def neighbors(self, sid: int) -> tuple[str, str]:
        """같은 칸 안의 앞뒤 문장(없으면 빈 글)."""
        same = [s for s in self.sentences if s.slot == self.get(sid).slot]
        i = next(k for k, s in enumerate(same) if s.id == sid)
        return (same[i - 1].text if i > 0 else "", same[i + 1].text if i + 1 < len(same) else "")


def injected_sentences(base_text: str, new_text: str) -> list[tuple[int, str]]:
    """바뀐 칸 글에서 새로 생기거나 바뀐 문장 [(바뀐 글 안 문장 차례, 문장)]. 문장이 둘로 나뉘면 둘 다.
    문장 순서만 바뀐 경우(같은 문장들, 다른 순서)는 자리가 바뀐 문장 모두(바꿈 차이로는 한쪽만 잡힌다)."""
    a, b = V.sentences(base_text), V.sentences(new_text)
    if a == b:
        return []
    if sorted(a) == sorted(b):
        return [(j, s) for j, s in enumerate(b) if a[j] != s]
    out = []
    for tag, _i1, _i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag in ("replace", "insert"):
            out += [(j, b[j]) for j in range(j1, j2)]
    return out


def _slot_piece_id(slot_path: str) -> str | None:
    """칸 글이 조각 글 그대로인 칸의 조각 id(권고표 칸은 '권고: … / 기대 효과: …' 로 섞여 None)."""
    for sid, (group, name) in NARRATIVE_PATHS.items():
        if slot_path == f"{group}.{name}":
            return f"slot:{sid}"
    for place, name in RECOMMEND_PATHS:
        if slot_path == f"recommendation_narrative.{name}":
            return f"slot:9:{place}"
    return None


def injected_ids(index: DocIndex, slot_path: str, injected: list[tuple[int, str]]) -> list[int]:
    """주입 문장 → 문서 문장 번호. 서술 칸은 조각 안 차례로(글이 같다), 권고표 칸은 같은 칸 문장 가운데 그 글을 담은 문장으로.
    찾지 못하면 ValueError(결함본·문서가 어긋남)."""
    pid = _slot_piece_id(slot_path)
    out: set[int] = set()
    if pid is not None:
        mine = [s for s in index.sentences if s.piece_id == pid]
        for j, text in injected:
            if j >= len(mine) or _norm(mine[j].text) != _norm(text):
                raise ValueError(f"주입 문장을 문서 조각 {pid} 의 {j + 1}번째 문장에서 찾지 못했다")
            out.add(mine[j].id)
        return sorted(out)
    slot = slot_of_path(slot_path)
    cands = [s for s in index.sentences if s.slot == slot]
    for _, text in injected:
        n = _norm(text)
        hit = ([s.id for s in cands if _norm(s.text) == n] or [s.id for s in cands if n and n in _norm(s.text)]
               or [s.id for s in cands if touches(text, s.text)])
        if not hit:
            raise ValueError(f"주입 문장을 칸 {slot} 에서 찾지 못했다")
        out.update(hit)
    return sorted(out)


# ── 판정자 표시(충실성 위반·일관성 모순) ─────────────────────────────────────────

FAITH_LABEL = {"violation": "사실 일치 위반", "inference_violation": "단정형 추론"}
CONTRADICTION_NAME = {"value_mismatch": "값 불일치", "verdict_vs_recommendation": "판정↔권고",
                      "weakness_vs_recommendation": "약점↔권고", "analysis_vs_finding": "분석↔소견"}
KIND_NAME = {"number": "숫자", "name": "이름", "comparison": "비교", "degree": "정도", "inference": "추론"}
RELATION_NAME = {"eq": "같음", "gt": "큼", "lt": "작음", "ge": "이상", "le": "이하"}
_FIELD_REF = re.compile(r"\{([^{}]+)\}")


def canon_field(name: str) -> str:
    name = str(name or "").strip()
    return ALIASES.get(name, name)


def answered_fields(detail: dict | None) -> list[str]:
    """에이전트가 답한 대상 — 식의 {칸}, 이름 칸, 지표 번호(metric.<id>.value)."""
    detail = detail or {}
    out: list[str] = []
    for key in ("expr", "expr_b"):
        out += [canon_field(m) for m in _FIELD_REF.findall(str(detail.get(key) or ""))]
    if str(detail.get("name_field") or "").strip():
        out.append(canon_field(detail["name_field"]))
    if str(detail.get("metric_id") or "").strip():
        out.append(f"metric.{detail['metric_id'].strip()}.value")
    return list(dict.fromkeys(out))


def _data_value(detail: dict | None):
    detail = detail or {}
    if "data" in detail:
        return detail["data"]
    if "a" in detail or "b" in detail:
        return f"{detail.get('a')} / {detail.get('b')}"
    return None


def flags_of(run: dict) -> list[dict]:
    """회차 결과 원본 → 표시 목록(표시 순번 index 1부터). 충실성 위반(violation·inference_violation)과 일관성 모순(인용 둘)."""
    rec = run.get("record") or {}
    claims = {c.get("id"): c for c in (rec.get("extract") or {}).get("claims") or []}
    out: list[dict] = []
    for v in (rec.get("faithfulness") or {}).get("verdicts") or []:
        if v.get("verdict") not in BLOCKING_VERDICTS:
            continue
        c = claims.get(v.get("claim_id"), {})
        detail = v.get("detail") or {}
        targets = answered_fields(detail)
        out.append({"index": len(out) + 1, "axis": FAITH, "kind": v.get("kind", ""), "verdict": v["verdict"],
                    "label": FAITH_LABEL[v["verdict"]], "quotes": [v.get("quote", "")], "pieces": [v.get("piece_id", "")],
                    "slots": [v.get("slot", "")], "claim_id": v.get("claim_id", ""),
                    "claim_fields": [f for f in (canon_field(c.get("field")), canon_field(c.get("field_b"))) if f],
                    "field": ", ".join(targets) or canon_field(c.get("field")), "data": _data_value(detail),
                    "detail": detail})
    for c in (rec.get("consistency") or {}).get("contradictions") or []:
        detail = c.get("detail") or {}
        kind = c.get("type", "")
        out.append({"index": len(out) + 1, "axis": CONS, "kind": kind, "verdict": "contradiction",
                    "label": f"모순({CONTRADICTION_NAME.get(kind, kind)})", "quotes": list(c.get("quotes") or []),
                    "pieces": list(c.get("pieces") or []), "slots": list(c.get("slots") or []), "claim_id": "",
                    "claim_fields": [], "field": canon_field(detail.get("field")), "data": None, "detail": detail})
    return out


def locate_flag(flag: dict, index: DocIndex) -> dict:
    """표시에 문장 번호를 붙인다 — 인용마다 문장들(quote_ids), 합친 것(sentence_ids)."""
    pieces = list(flag.get("pieces") or [])
    per = [index.locate(q, pieces[i] if i < len(pieces) else "") for i, q in enumerate(flag.get("quotes") or [])]
    return {**flag, "quote_ids": per, "sentence_ids": sorted({i for ids in per for i in ids})}


def run_ok(run: dict | None) -> bool:
    """판정이 난 회차(게이트 pass·blocked). 실패 회차(게이트 없음)는 아니다."""
    return bool(run) and run.get("gate") in ("pass", "blocked")


# ── 6.2 탐지 ────────────────────────────────────────────────────────────


def detect(run: dict | None, index: DocIndex, injected: list[int], expected_axes) -> dict:
    """결함본 한 회차의 탐지. 게이트 수준 = 차단 표시 어느 것이든 주입 문장에, 축 수준 = 기대 축마다 그 축의 표시가 주입 문장에.
    위치 불일치 = 주입 문장에 닿지 않은 표시(과잉 차단 판정 대상). 실패·미실행 회차는 탐지 값이 None."""
    if run is None:
        return {"status": NOT_RUN, "gate": None, "failure": None, "gate_detected": None,
                "axes": {a: None for a in expected_axes}, "at_injected": [], "out_of_place": []}
    if not run_ok(run):
        return {"status": FAILED, "gate": None, "failure": run.get("failure") or run.get("error"), "gate_detected": None,
                "axes": {a: None for a in expected_axes}, "at_injected": [], "out_of_place": []}
    if not injected:
        raise ValueError("주입 문장이 없다 — 결함본 meta 를 확인한다")
    hit = set(injected)
    located = [locate_flag(f, index) for f in flags_of(run)]
    at = [f for f in located if hit & set(f["sentence_ids"])]
    out = [f for f in located if not hit & set(f["sentence_ids"])]
    return {"status": DETECTED if at else MISSED, "gate": run["gate"], "failure": None, "gate_detected": bool(at),
            "axes": {a: any(f["axis"] == a for f in at) for a in expected_axes},
            "at_injected": at, "out_of_place": out}


def word(value: bool | None, status: str | None = None) -> str:
    if value is None:
        return status or FAILED
    return DETECTED if value else MISSED


def two_run_class(values: list[bool | None]) -> str:
    """회차별 탐지 값 둘(True 탐지·False 미탐지·None 실패) → 묶음 분류. 실패는 미탐지로 세지 않고 따로 적는다."""
    if len(values) != 2:
        return f"{NOT_RUN}(회차 {len(values)}개)"
    hits = [v for v in values if v is not None]
    fails = 2 - len(hits)
    if fails == 2:
        return "두 회차 모두 실패"
    if fails == 1:
        return f"한 회차 {DETECTED if hits[0] else MISSED} · 한 회차 실패"
    if all(hits):
        return BOTH
    if any(hits):
        return ONE
    return NEITHER


def wilson(k: int, n: int) -> dict:
    """Wilson 95% 구간(기존 wilson_interval). n=0 이면 구간 없음."""
    if n <= 0:
        return {"k": k, "n": n, "rate": None, "low": None, "high": None}
    low, high = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n, "low": low, "high": high}


# ── 6.3 미탐지 원인 ──────────────────────────────────────────────────────

CAUSES = ("추출 누락", "필드 오지정", "대상 오답", "규칙 허용", "모순 미검출")
UNCLASSIFIED = "분류 불가"


def claims_at(run: dict, index: DocIndex, sentence_ids) -> list[dict]:
    """주입 문장(문장 번호들)에서 추출된 주장 — 주장 인용을 문장에 붙여(6.1 대조) 겹치는 것."""
    want = set(sentence_ids)
    out = []
    for c in ((run.get("record") or {}).get("extract") or {}).get("claims") or []:
        ids = index.locate(c.get("quote", ""), c.get("piece_id", ""))
        if want & set(ids):
            out.append({**c, "sentence_ids": ids})
    return out


def _expected_fields(fields) -> set[str]:
    if fields is None:
        return set()
    if isinstance(fields, str):
        fields = [f for f in re.split(r"[|,]", fields)]
    return {canon_field(f) for f in fields if str(f).strip()}


def miss_cause(defect_id: str, run: dict, index: DocIndex, injected: list[int], expected_fields=None) -> dict:
    """놓친 결함 한 회차의 원인(6.3) — 파이프라인 순서상 가장 앞선 것과 그 근거(주입 문장의 주장·필드·판정).
    expected_fields = 결함 meta 의 근거 필드(grounds.field — 문자열 'a|b' 또는 목록). 비었으면 2·3단계를 판단할 수 없어 건너뛴다."""
    if defect_id not in RELATED_KINDS:
        raise ValueError(f"평가 결함이 아니다: {defect_id}")
    rec = run.get("record") or {}
    claims = claims_at(run, index, injected)
    verdicts = {v.get("claim_id"): v for v in (rec.get("faithfulness") or {}).get("verdicts") or []}
    contradictions = [{"type": c.get("type"), "quotes": c.get("quotes")} for c in (rec.get("consistency") or {}).get("contradictions") or []]

    def evidence(cs):
        return [{"claim_id": c.get("id"), "kind": c.get("kind"), "quote": c.get("quote"), "field": c.get("field"),
                 "field_b": c.get("field_b"), "verdict": (verdicts.get(c.get("id")) or {}).get("verdict"),
                 "detail": (verdicts.get(c.get("id")) or {}).get("detail"),
                 "answered_fields": answered_fields((verdicts.get(c.get("id")) or {}).get("detail"))} for c in cs]

    def result(order: int | None, note: str = "", cs=None) -> dict:
        return {"cause": CAUSES[order - 1] if order else UNCLASSIFIED, "order": order, "note": note,
                "claims": evidence(claims if cs is None else cs), "contradictions": contradictions}

    kinds = RELATED_KINDS[defect_id]
    if kinds is None:  # C02~C09 — 일관성 경로, 5번만
        return result(5)
    related = [c for c in claims if c.get("kind") in kinds]
    if not related:
        return result(1, f"주입 문장에서 {'·'.join(kinds)} 주장이 추출되지 않음")
    wanted = _expected_fields(expected_fields)
    note = "" if wanted else "근거 필드 없음 — 필드 오지정·대상 오답은 판단하지 않음"
    if wanted:
        aimed = [c for c in related if {canon_field(c.get("field")), canon_field(c.get("field_b"))} & wanted]
        if not aimed:
            return result(2, f"추출 주장의 대상 필드가 근거 필드({', '.join(sorted(wanted))})가 아님", related)
        right = [c for c in aimed if set(answered_fields((verdicts.get(c.get("id")) or {}).get("detail"))) & wanted]
        if not right:
            return result(3, "에이전트가 답한 대상이 근거 필드와 다름", aimed)
    else:
        right = related
    if not any((verdicts.get(c.get("id")) or {}).get("verdict") in BLOCKING_VERDICTS for c in right):
        return result(4, note or "주장·대상은 맞았으나 코드 규칙이 위반으로 보지 않음", right)
    if CONS in PREREG_DEFECTS[defect_id].expected_axes:
        return result(5, note, right)
    return result(None, "주입 문장의 위반 판정이 있는데 탐지되지 않음(인용 대조 확인 필요)", right)


# ── 6.6 불변성·6.7 독립성 ─────────────────────────────────────────────────


def within(value, base_values: list, axis: str) -> tuple[bool | None, str]:
    """점수 하나가 원본 회차들의 [최솟값 − 허용 폭, 최댓값 + 허용 폭] 안인가. None 처리(명시):
    모두 None(분모 0·없음) → 같다(True, '모두 점수 없음'), 일부만 None → 판단 불가(None)."""
    vals = [*base_values, value]
    if all(v is None for v in vals):
        return True, "모두 점수 없음"
    if any(v is None for v in vals):
        return None, "점수 없음이 섞임"
    tol = TOLERANCE[axis]
    lo, hi = min(base_values) - tol, max(base_values) + tol
    return lo - FLOAT_EPS <= value <= hi + FLOAT_EPS, f"[{lo:.4g}, {hi:.4g}]"


def _scores(run: dict | None) -> dict:
    return axis_scores(run) if run else {a: None for a in AXES}


def _tri(values) -> bool | None:
    values = list(values)
    if any(v is False for v in values):
        return False
    if any(v is None for v in values):
        return None
    return True


def invariance(base_runs: list[dict | None], variant_runs: list[dict | None]) -> dict:
    """변형 하나의 불변성(6.6). 두 회차 모두 발급 결정이 원본과 같고 축 점수가 원본 범위(허용 폭) 안이면 불변.
    원본 두 회차의 결정이 갈렸으면 결정은 '판단 불가(원본 갈림)'."""
    out: dict = {"base_gates": [r.get("gate") if r else None for r in base_runs],
                 "variant_gates": [r.get("gate") if r else None for r in variant_runs]}
    if len(base_runs) != 2 or len(variant_runs) != 2 or not all(base_runs) or not all(variant_runs):
        return {**out, "status": "판단 불가(회차 부족)", "decision": None, "axes": {}, "invariant": None}
    if not all(run_ok(r) for r in base_runs):
        return {**out, "status": "판단 불가(원본 실패)", "decision": None, "axes": {}, "invariant": None}
    base_scores = [_scores(r) for r in base_runs]
    axes = {}
    for axis in AXES:
        per = [within(_scores(r)[axis] if run_ok(r) else None, [s[axis] for s in base_scores], axis) for r in variant_runs]
        axes[axis] = {"base": [s[axis] for s in base_scores], "variant": [_scores(r)[axis] if run_ok(r) else None for r in variant_runs],
                      "ok": _tri(ok for ok, _ in per), "notes": [n for _, n in per]}
    if not all(run_ok(r) for r in variant_runs):
        return {**out, "status": "판단 불가(변형 실패)", "decision": None, "axes": axes, "invariant": None}
    if len(set(out["base_gates"])) != 1:
        return {**out, "status": "판단 불가(원본 갈림)", "decision": None, "axes": axes, "invariant": None}
    decision = all(g == out["base_gates"][0] for g in out["variant_gates"])
    inv = _tri([decision, *(a["ok"] for a in axes.values())])
    status = {True: "불변", False: "불변 아님", None: "판단 불가(점수 없음 섞임)"}[inv]
    return {**out, "status": status, "decision": decision, "axes": axes, "invariant": inv}


def independence(base_runs: list[dict | None], defect_runs: list[dict | None], expected_axes, allowed_secondary) -> dict:
    """결함 하나의 독립성 표(6.7) — 기대 축이 아닌 축마다 결함본 회차 점수가 원본 두 회차 범위(허용 폭) 안인가.
    허용 부차 축에서 벗어난 것은 '허용'."""
    ok_base = [r for r in base_runs if run_ok(r)]
    rows = {}
    for axis in AXES:
        if axis in expected_axes:
            continue
        base_vals = [_scores(r)[axis] for r in ok_base]
        vals = [_scores(r)[axis] if run_ok(r) else None for r in defect_runs]
        if len(ok_base) != 2 or not defect_runs or not all(run_ok(r) for r in defect_runs):
            rows[axis] = {"base": base_vals, "defect": vals, "ok": None, "status": "판단 불가(실패·회차 부족)"}
            continue
        per = [within(v, base_vals, axis) for v in vals]
        ok = _tri(o for o, _ in per)
        status = {True: "범위 안", None: "판단 불가(점수 없음 섞임)"}.get(ok) if ok is not False else (
            "허용" if axis in allowed_secondary else "벗어남")
        rows[axis] = {"base": base_vals, "defect": vals, "ok": ok, "status": status, "notes": [n for _, n in per]}
    return rows


# ── 6.5 재현성 ──────────────────────────────────────────────────────────


def flagged_set(run: dict, index: DocIndex) -> set[int]:
    """검출 문장 집합 — 차단 표시(충실성 위반·일관성 모순)가 붙은 문장 번호."""
    return {i for f in flags_of(run) for i in locate_flag(f, index)["sentence_ids"]}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def reproducibility(per_doc: dict[str, list[dict | None]], indexes: dict[str, DocIndex]) -> dict:
    """문서마다 두 회차 → 발급 결정 일치 비율, 검출 문장 집합 완전 일치 비율과 평균 Jaccard(둘 다 비면 1),
    '두 회차 모두 통과 / 두 회차 모두 차단 / 갈림'으로 나눠. 실패·미실행 회차가 있는 문서는 비율에서 빼고 따로 적는다."""
    docs, excluded = {}, {}
    for doc, runs in sorted(per_doc.items()):
        if len(runs) != 2 or not all(run_ok(r) for r in runs):
            excluded[doc] = [r.get("gate") if r else None for r in runs]
            continue
        sets = [flagged_set(r, indexes[doc]) for r in runs]
        gates = [r["gate"] for r in runs]
        group = ("두 회차 모두 통과" if gates == ["pass", "pass"] else "두 회차 모두 차단" if gates == ["blocked", "blocked"]
                 else "갈림")
        docs[doc] = {"gates": gates, "group": group, "same_decision": gates[0] == gates[1],
                     "sets": [sorted(s) for s in sets], "exact": sets[0] == sets[1], "jaccard": jaccard(*sets)}

    def summary(rows: list[dict]) -> dict:
        n = len(rows)
        return {"n": n, "decision_agreement": sum(r["same_decision"] for r in rows) / n if n else None,
                "exact_rate": sum(r["exact"] for r in rows) / n if n else None,
                "mean_jaccard": sum(r["jaccard"] for r in rows) / n if n else None}

    groups = {g: summary([r for r in docs.values() if r["group"] == g]) for g in ("두 회차 모두 통과", "두 회차 모두 차단", "갈림")}
    return {"overall": summary(list(docs.values())), "groups": groups, "docs": docs, "excluded": excluded}


# ── 6.8 주입 문장 추출률 ───────────────────────────────────────────────────


def extraction_hit(defect_id: str, run: dict | None, index: DocIndex, injected: list[int]) -> bool | None:
    """주입 문장에서 관련 종류(C02~C09 는 어떤 종류든) 주장이 추출됐나. 추출 기록이 없는 회차(실패·미실행)는 None."""
    if not run or "extract" not in (run.get("record") or {}) or run["record"]["extract"].get("failure"):
        return None
    kinds = RELATED_KINDS[defect_id]
    claims = claims_at(run, index, injected)
    return any(kinds is None or c.get("kind") in kinds for c in claims)


def rate(values: list[bool | None]) -> dict:
    known = [v for v in values if v is not None]
    return {"k": sum(known), "n": len(known), "rate": sum(known) / len(known) if known else None,
            "no_record": len(values) - len(known)}


# ── 6.9 숫자 토큰 재현율 ──────────────────────────────────────────────────

NUMBER_TOKEN = re.compile(r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(%?)")  # §6.9 정규식과 뒤따르는 %
_CI_AFTER = re.compile(r"\s*%?\s*신뢰\s*구간")
_SECTION_AFTER = re.compile(r"절")  # 붙여 쓴 절 번호('4.4절')
_TABLE_BEFORE = re.compile(r"(?<![가-힣])표\s*$")  # '표 3' — '목표 0.900' 은 아니다
_REPEAT_AFTER = re.compile(r"[회번]")


def numeric_tokens(text: str, *, exclude: bool = True) -> list[str]:
    """숫자 토큰(§6.9). 제외: 영문자에 붙은 숫자(F1·Class9·M4), '95% 신뢰구간'의 95, 절·표 번호('4.4절'·'표 3'),
    뒤에 '회'·'번'이 붙은 수(반복 횟수 같은 방법 서술)."""
    out = []
    for m in NUMBER_TOKEN.finditer(text or ""):
        number, pct = m.group(1), m.group(2)
        if exclude:
            before = text[m.start() - 1] if m.start() > 0 else ""
            after_number = text[m.end(1)] if m.end(1) < len(text) else ""
            if re.match(r"[A-Za-z]", before) or re.match(r"[A-Za-z]", after_number):
                continue
            if _CI_AFTER.match(text, m.end()):
                continue
            if _SECTION_AFTER.match(text, m.end()) or _TABLE_BEFORE.search(text[:m.start()]):
                continue
            if _REPEAT_AFTER.match(text, m.end()):
                continue
        out.append(number + pct)
    return out


def numeric_recall(run: dict, index: DocIndex) -> dict:
    """원본 한 회차 — 서술 문장의 숫자 토큰마다 같은 문장의 숫자 주장 인용에 그 토큰이 들어 있으면 대응(필드는 보지 않음)."""
    claims = [c for c in ((run.get("record") or {}).get("extract") or {}).get("claims") or [] if c.get("kind") == "number"]
    by_sentence: dict[int, list[str]] = {}
    for c in claims:
        for sid in index.locate(c.get("quote", ""), c.get("piece_id", "")):
            by_sentence.setdefault(sid, []).extend(numeric_tokens(c.get("quote", ""), exclude=False))
    total, matched, missed = 0, 0, []
    for s in index.sentences:
        for tok in numeric_tokens(s.text):
            total += 1
            if tok in by_sentence.get(s.id, []):
                matched += 1
            else:
                missed.append({"sentence_id": s.id, "token": tok})
    return {"tokens": total, "matched": matched, "recall": matched / total if total else None, "missed": missed}


# ── 5 베이스라인 ───────────────────────────────────────────────────────────


def baseline1_detection(findings: list[dict], injected: list[int], injected_sections) -> dict:
    """베이스라인 1(1차 점검) 결함본 탐지 — 검출 문장이 주입 문장이면 탐지, 칸 단위로만 나온 검출(문장 없음)이 주입 문장의
    칸에 있으면 '칸 단위 검출'(따로 표시). findings = [{section, check, value, sentence_ids}]."""
    hit = set(injected)
    at = [f for f in findings if hit & set(f.get("sentence_ids") or [])]
    section_only = [f for f in findings if not f.get("sentence_ids") and f.get("section") in set(injected_sections)]
    status = DETECTED if at else ("칸 단위 검출" if section_only else MISSED)
    return {"status": status, "detected": bool(at), "at_injected": at, "section_only": section_only,
            "elsewhere": [f for f in findings if f not in at and f not in section_only]}


def baseline2_issues(result: dict | None, index: DocIndex) -> list[dict]:
    """베이스라인 2 지적 → 문장 번호를 붙인 목록(조각을 모르므로 문서 전체에서 찾는다)."""
    out = []
    for i, issue in enumerate((result or {}).get("issues") or [], 1):
        quotes = [q for q in issue.get("quotes") or [] if isinstance(q, str)]
        per = [index.locate(q) for q in quotes]
        out.append({"index": i, "axis": issue.get("axis", ""), "type": issue.get("type", ""), "quotes": quotes,
                    "reason": issue.get("reason", ""), "quote_ids": per, "sentence_ids": sorted({x for ids in per for x in ids})})
    return out


def baseline2_detection(result: dict | None, index: DocIndex, injected: list[int]) -> dict:
    """베이스라인 2 결함본 탐지 — 인용 중 하나가 주입 문장에 대조되면 탐지(축은 보지 않는다, 게이트 수준만)."""
    if result is None:
        return {"status": NOT_RUN, "detected": None, "issues": 0, "decision": None}
    issues = baseline2_issues(result, index)
    hit = set(injected)
    at = [i for i in issues if hit & set(i["sentence_ids"])]
    return {"status": DETECTED if at else MISSED, "detected": bool(at), "issues": len(issues),
            "decision": result.get("decision"), "at_injected": at, "elsewhere": [i for i in issues if i not in at]}


# ── 6.4 과잉 차단 판정표 ───────────────────────────────────────────────────

SHEET_COLUMNS = ("문항", "표시 종류", "표시된 문장", "짝 문장(모순일 때)", "앞 문장", "뒤 문장", "판정자 사유", "대상 필드", "필드 값",
                 "판정(실제 오류/판정자 오판)", "판정 사유(필드 값과 함께 한 줄)")  # 판정표_양식.md 와 글자까지 같다
KEY_COLUMNS = ("문항", "실행 번호", "문서", "출처", "회차", "표시 순번", "축", "표시 종류")
VERDICT_COLUMN, VERDICT_REASON_COLUMN = SHEET_COLUMNS[9], SHEET_COLUMNS[10]
REAL_ERROR, JUDGE_ERROR = "실제 오류", "판정자 오판"
TEXT_COLUMNS = SHEET_COLUMNS[2:6]  # 문서 글이 들어가는 칸
B2_LABEL = "지적"  # 베이스라인 지적은 종류를 '지적'으로만(판정표_양식.md)
OUTSIDE = "(서술 밖 인용)"


def _sentence_texts(ids: list[int], index: DocIndex) -> list[str]:
    return [index.get(i).text for i in ids]


def _value_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _reason(flag: dict) -> str:
    """판정자 사유 — 주장 종류, 관계, 판정자가 고른 대상과 그 근거 값(판정표_양식.md). 주장 번호·실행 번호는 싣지 않는다."""
    if flag["axis"] == CONS:
        name = CONTRADICTION_NAME.get(flag["kind"], flag["kind"])
        if flag["kind"] == "value_mismatch":
            return f"모순 유형 {name}; 같은 양으로 묶인 대상 {flag['field'] or '-'}; 두 값 {' · '.join(flag['quotes'])}"
        return f"모순 유형 {name}"
    d = flag.get("detail") or {}
    parts = [f"주장 종류 {KIND_NAME.get(flag['kind'], flag['kind'])}"]
    if flag["kind"] == "inference" and "marked" in d:
        parts.append(f"추론 표시 {'있음' if d.get('marked') else '없음'}")
        if d.get("causal"):
            parts.append("인과 낱말 있음")
        if d.get("basis"):
            parts.append(f"근거 갈래 {d['basis']}")
    if d.get("relation"):
        parts.append(f"관계 {RELATION_NAME.get(d['relation'], d['relation'])}")
    if flag.get("field"):
        parts.append(f"대상 {flag['field']}")
    if flag.get("data") is not None:
        parts.append(f"근거 값 {_value_text(flag['data'])}")
    return "; ".join(parts)


def sheet_entries(run_id: str, doc: str, source: str, round_no, located: list[dict], index: DocIndex,
                  value_of=None) -> list[dict]:
    """한 회차의 (문장 번호가 붙은) 표시 → 판정표 후보(묶기 전). value_of(필드) → 데이터 값(값 불일치의 필드 값 칸)."""
    out = []
    for f in located:
        per = f.get("quote_ids") or [f.get("sentence_ids") or []]
        first = per[0] if per else []
        texts = _sentence_texts(first, index) or [f"{OUTSIDE} {f['quotes'][0]}" if f.get("quotes") else OUTSIDE]
        pair = ""
        if f["axis"] == CONS and len(f.get("quotes") or []) > 1:
            second = per[1] if len(per) > 1 else []
            pair = " … ".join(_sentence_texts(second, index)) or f"{OUTSIDE} {f['quotes'][1]}"
        prev, nxt = index.neighbors(first[0]) if first else ("", "")
        value = f.get("data")
        if value is None and f.get("field") and value_of is not None:
            value = value_of(f["field"])
        out.append({"run_id": run_id, "doc": doc, "source": source, "round": round_no, "index": f["index"], "axis": f["axis"],
                    "label": f["label"], "text": " … ".join(texts), "pair": pair, "prev": prev, "next": nxt,
                    "reason": _reason(f), "field": f.get("field") or "", "value": _value_text(value)})
    return out


def b2_entries(run_id: str, doc: str, source: str, issues: list[dict], index: DocIndex) -> list[dict]:
    """베이스라인 2 지적 → 판정표 후보. 종류는 '지적'으로만, 사유는 지적 글 그대로."""
    out = []
    for i in issues:
        per = i["quote_ids"]
        first = per[0] if per else []
        texts = _sentence_texts(first, index) or [f"{OUTSIDE} {i['quotes'][0]}" if i["quotes"] else OUTSIDE]
        pair = " … ".join(_sentence_texts(per[1], index)) if len(per) > 1 else ""
        prev, nxt = index.neighbors(first[0]) if first else ("", "")
        out.append({"run_id": run_id, "doc": doc, "source": source, "round": "", "index": i["index"], "axis": i["axis"],
                    "label": B2_LABEL, "text": " … ".join(texts), "pair": pair, "prev": prev, "next": nxt,
                    "reason": i["reason"], "field": "", "value": ""})
    return out


def build_sheet(entries: list[dict], seed: int) -> tuple[list[tuple], list[tuple], list[dict]]:
    """(판정표 줄, 열쇠 줄, 묶음). 같은 문장(글자 같음)·같은 표시 종류·같은 대상 필드의 표시는 한 문항(모순은 두 문장 짝 순서 무관).
    문항 번호 J001… 은 시드로 섞은 차례."""
    groups: dict[tuple, list[dict]] = {}
    for e in entries:
        sentences = tuple(sorted([e["text"], e["pair"]])) if e["pair"] else (e["text"],)
        groups.setdefault((sentences, e["label"], e["field"]), []).append(e)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    rows, key_rows, items = [], [], []
    for n, k in enumerate(keys, 1):
        qid = f"J{n:03d}"
        members = groups[k]
        head = members[0]
        reasons = list(dict.fromkeys(m["reason"] for m in members if m["reason"]))
        values = list(dict.fromkeys(m["value"] for m in members if m["value"]))
        rows.append((qid, head["label"], head["text"], head["pair"], head["prev"], head["next"], " / ".join(reasons),
                     head["field"], " / ".join(values), "", ""))
        for m in members:
            key_rows.append((qid, m["run_id"], m["doc"], m["source"], m["round"], m["index"], m["axis"], m["label"]))
        items.append({"item": qid, "members": members})
    return rows, key_rows, items


# 판정표에 새면 안 되는 것(판정표_양식.md "싣지 않는 것")
LEAK_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("결함·변형 ID", re.compile(r"[FCV]\d{2}")),
    ("실행 번호", re.compile(r"[A-Z]-\d{4}")),
    ("회차", re.compile(r"[12]\s*회차|eval_r\d|(?<![A-Za-z])r[12](?![0-9A-Za-z])|(?<![A-Za-z])round(?![A-Za-z])", re.IGNORECASE)),
)
SOURCE_WORDS = re.compile(r"원본|변형|결함본|베이스라인|sham|(?<![A-Za-z_])base(?![A-Za-z_])", re.IGNORECASE)


def sheet_leaks(rows: list[tuple], *, doc_names=(), allowed_texts=()) -> list[str]:
    """판정표 줄에서 새는 것 — 문서 이름·결과서 태그·결함/변형 ID·실행 번호·회차는 모든 칸에서, 출처 낱말(원본·변형·결함본·
    베이스라인·base)은 문서 글·데이터 값(allowed_texts)을 걷어 낸 나머지에서. 빈 목록이면 새는 것 없음."""
    names = sorted({*REPORT_TAGS, *(n for n in doc_names if n)}, key=len, reverse=True)
    allowed = sorted({t for t in allowed_texts if t}, key=len, reverse=True)
    problems = []
    for r, row in enumerate(rows, 1):
        for c, cell in enumerate(row):
            text = str(cell or "")
            col = SHEET_COLUMNS[c] if c < len(SHEET_COLUMNS) else str(c)
            for name in names:
                if name.lower() in text.lower():
                    problems.append(f"{r}번째 줄 '{col}': 문서 이름·결과서 태그 {name!r}")
            for what, pat in LEAK_PATTERNS:
                m = pat.search(text)
                if m:
                    problems.append(f"{r}번째 줄 '{col}': {what} {m.group(0)!r}")
            rest = text
            for t in allowed:
                rest = rest.replace(t, " ")
            m = SOURCE_WORDS.search(rest)
            if m:
                problems.append(f"{r}번째 줄 '{col}': 출처 낱말 {m.group(0)!r}")
    return problems


def tally_overblocking(key_rows: list[dict], answers: dict[str, str], runs: dict[str, dict]) -> dict:
    """판정 뒤 셈(6.4). key_rows = 열쇠 줄(dict), answers = {문항: 실제 오류|판정자 오판|''},
    runs = {실행 번호: {doc, source, gate, checked(충실성 검사한 주장 수)}} — 원본·변형·결함본 회차.
    차단된 원본·변형 회차: 표시가 모두 판정됐고 '판정자 오판'만 있으면 과잉 차단, '실제 오류'가 있으면 자연 발생 오류 검출,
    덜 판정됐으면 판정 미완. 주장 단위: 원본·변형의 '판정자 오판' 충실성 위반 수 / 원본·변형 충실성 checked 합."""
    bad = {v for v in answers.values() if v not in (REAL_ERROR, JUDGE_ERROR, "")}
    if bad:
        raise ValueError(f"판정은 '{REAL_ERROR}' 또는 '{JUDGE_ERROR}': {sorted(bad)}")
    flags_by_run: dict[str, list[tuple[dict, str]]] = {}
    for k in key_rows:
        flags_by_run.setdefault(k["실행 번호"], []).append((k, answers.get(k["문항"], "")))
    per_run, counts = {}, {"과잉 차단": 0, "자연 발생 오류 검출": 0, "판정 미완": 0, "차단": 0, "통과": 0}
    wrong_faith, checked = 0, 0
    out_of_place = {REAL_ERROR: 0, JUDGE_ERROR: 0, "": 0}
    b2 = {REAL_ERROR: 0, JUDGE_ERROR: 0, "": 0}
    for k, v in (pair for pairs in flags_by_run.values() for pair in pairs):
        if k["출처"] == SOURCE_DEFECT:
            out_of_place[v] += 1
        elif k["출처"] == SOURCE_B2:
            b2[v] += 1
    for run_id, info in sorted(runs.items()):
        if info["source"] not in (SOURCE_ORIGINAL, SOURCE_VARIANT) or info.get("gate") not in ("pass", "blocked"):
            continue
        checked += int(info.get("checked") or 0)
        flags = flags_by_run.get(run_id, [])
        wrong_faith += sum(1 for k, v in flags if v == JUDGE_ERROR and k["축"] == FAITH)
        if info["gate"] == "pass":
            counts["통과"] += 1
            continue
        counts["차단"] += 1
        verdicts = [v for _, v in flags]
        if REAL_ERROR in verdicts:
            status = "자연 발생 오류 검출"
        elif verdicts and all(v == JUDGE_ERROR for v in verdicts):
            status = "과잉 차단"
        else:
            status = "판정 미완"
        counts[status] += 1
        per_run[run_id] = {"doc": info["doc"], "source": info["source"], "status": status,
                           "verdicts": {REAL_ERROR: verdicts.count(REAL_ERROR), JUDGE_ERROR: verdicts.count(JUDGE_ERROR),
                                        "미판정": verdicts.count("")}}
    return {"runs": per_run, "counts": counts,
            "claim_level": {"wrong_faithfulness_violations": wrong_faith, "checked": checked,
                            "rate": wrong_faith / checked if checked else None},
            "defect_out_of_place": {"실제 오류": out_of_place[REAL_ERROR], "판정자 오판": out_of_place[JUDGE_ERROR],
                                    "미판정": out_of_place[""]},
            "baseline2": {"실제 오류": b2[REAL_ERROR], "판정자 오판": b2[JUDGE_ERROR], "미판정": b2[""]}}
