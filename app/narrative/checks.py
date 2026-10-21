"""app/narrative/checks.py — 칸별 검사(서술 설계 D36 검문 사다리의 '검사', 계획 설계 요지 5).

AI 가 쓴 칸 하나를 다섯 가지로 본다. 걸린 칸만 다시 만들고, 또 걸리면 그 칸만 규칙 템플릿으로 바꾸는 처리는
app.narrative.generator 가 한다. 검사는 순수 함수다(호출·기록 없음).

① 숫자: 그 칸 글의 숫자가 모두 허용 목록 안(app.narrative.grounding — 원본 사실 시트로 만든 목록).
② 판정 모순: 강제된 판정과 어긋나는 문장(grounding.find_verdict_contradictions — 오탐·누락을 함께 고친 판, 점검 BE-04).
③ 9칸: 기대 효과의 예측 수치(지표값·향상폭 — 사용자 결정 "막는 숫자는 예측 수치만", 다-9: '1차 원인'·'3개월 내'는
   막지 않는다), 권고가 빠진 문제(D22 — 문제마다 권고 하나 이상).
④ D24 선언 숫자: 그 칸의 D24 항목(7.1 위험 쪽 오류·8.2 선언 평균 방식·8.3 대화형 응답 시간)의 확인 숫자 중 하나가
   글에 있는가(required_items.Conclusion.d24_values, 허용 목록과 같은 표기 변형). '특별한 것 없음'이면 항목이 없어 끈다.
⑤ 형식: 빈 칸, 7.3 은 허용 숫자를 하나 이상 인용(D7·D38), 영어 판정 코드·필드명(작성 규칙 D37 — 운영에서 "판정은
   FAIL이다"가 나왔다). 클래스·지표·고객 입력의 이름 속 밑줄 이름은 이름이라 뺀다.

상호작용
- 의존(import): app.narrative.grounding, app.narrative.required_items(Conclusion), app.narrative.schemas
- 사용처: app.narrative.generator, tests
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.narrative.grounding import (
    _CTX_TOKEN_RE,
    _NUMBER_RE,
    _add,
    _canon,
    _normalize_minus,
    find_verdict_contradictions,
    verify_grounding,
)
from app.narrative.required_items import Conclusion
from app.narrative.schemas import FactSheet

# 칸마다 AI 응답에서 검사할 글(응답 필드 점 표기). 9칸은 권고표의 글도 본다.
REQUIRED_NON_EMPTY = ("7.1", "7.2", "8.1", "8.2", "8.3", "9")


@dataclass
class SectionCheck:
    section: str
    numbers: list[str] = field(default_factory=list)        # ① 허용 목록 밖 숫자
    contradiction: bool = False                              # ② 판정 모순
    impact_numbers: list[str] = field(default_factory=list)  # ③ 기대 효과의 예측 수치
    missing_recommendations: list[str] = field(default_factory=list)  # ③ 권고가 없는 문제 번호
    d24_missing: bool = False                                # ④ 선언 숫자 빠짐
    format: list[str] = field(default_factory=list)          # ⑤ empty · no_citation · code_words
    code_words: list[str] = field(default_factory=list)      # ⑤ 걸린 영어 코드·필드명

    @property
    def hard_failures(self) -> list[str]:
        """다시 만들거나 템플릿으로 바꿔야 하는 사유(D24 빠짐은 따로 — 다시 만든 뒤에는 그대로 싣는다)."""
        out = []
        if self.numbers:
            out.append("numbers")
        if self.contradiction:
            out.append("verdict_contradiction")
        if self.impact_numbers:
            out.append("impact_number")
        if self.missing_recommendations:
            out.append("missing_recommendation")
        out.extend(self.format)
        return out

    @property
    def passed(self) -> bool:
        return not self.hard_failures and not self.d24_missing

    def as_record(self) -> dict:
        return {
            "numbers": self.numbers,
            "verdict_contradiction": self.contradiction,
            "impact_numbers": self.impact_numbers,
            "missing_recommendations": self.missing_recommendations,
            "d24_missing": self.d24_missing,
            "format": self.format,
            "code_words": self.code_words,
        }


def section_texts(section_id: str, parts: dict) -> list[str]:
    """칸 하나의 글(검사 대상) — 응답 필드(점 표기) → 값에서."""
    if section_id == "9":
        texts = [parts.get("recommendation_narrative.data_quality", ""), parts.get("recommendation_narrative.model_ops", "")]
        for rec in parts.get("recommendations") or []:
            texts.extend([rec.get("category", ""), rec.get("action", ""), rec.get("expected_impact", "")])
        return texts
    path = {
        "7.1": "interpretation.confusion_analysis",
        "7.2": "interpretation.distribution_analysis",
        "7.3": "interpretation.additional_observations",
        "8.1": "conclusion.benchmark",
        "8.2": "conclusion.narrative",
        "8.3": "conclusion.risks",
    }[section_id]
    return [parts.get(path, "")]


# ── ③ 기대 효과의 예측 수치 ─────────────────────────────────────────────────────
# 숫자 뒤에 차례·기간·횟수 단위가 오면 예측 수치가 아니다('1차 원인', '3개월 내', '2회 재평가'). 문제 번호(P1)는 앞 글자가
# 영문이라 숫자로 읽지 않는다. 소수·백분율·%p 는 단위와 무관하게 예측 수치다(지표값·향상폭).
# 차례·기간·횟수·개수 단위. 단 단위 바로 뒤에 개선을 말하는 말이 붙으면("3시간 단축"·"5회 감소") 향상폭 예측이다
# (S5 검토 checks-10: '개'·'분'이 빠져 "상위 3개 클래스"·"10분 내"가 막혔고, "3시간 단축"은 통과했다).
_NOT_PREDICTION_UNIT = re.compile(r"\s*(?:차|단계|번째|순위|순차|개월|주|일|년|분기|시간|회|번|분|개|건의\s*사례)")
_IMPROVEMENT_AFTER = re.compile(r"\s*(?:을|를|이|가|만큼|씩|정도)?\s*(?:단축|감소|줄|늘|향상|개선|증가|상승|낮아|높아|올라|내려)")


def predicted_numbers(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    stripped = _CTX_TOKEN_RE.sub(" ", _normalize_minus(text))
    found = []
    for m in _NUMBER_RE.finditer(stripped):
        token = m.group(0)
        if _canon(token) is None:
            continue
        decimal_or_pct = "." in token or "%" in token or stripped[m.end():m.end() + 2].startswith("%p")
        unit = None if decimal_or_pct else _NOT_PREDICTION_UNIT.match(stripped, m.end())
        if unit and not _IMPROVEMENT_AFTER.match(stripped, unit.end()):
            continue
        found.append(token)
    return found


# ── ④ D24 선언 숫자 ─────────────────────────────────────────────────────────────


def d24_present(text: str, values: tuple[float, ...]) -> bool:
    """글에 확인 숫자 중 하나가 허용 목록과 같은 표기 변형(소수 1~4자리·백분율)으로 있는가."""
    wanted: set = set()
    for v in values:
        _add(wanted, v)
        _add(wanted, abs(v))
    stripped = _CTX_TOKEN_RE.sub(" ", _normalize_minus(text or ""))
    return any(_canon(tok) in wanted for tok in _NUMBER_RE.findall(stripped))


# ── ⑤ 형식 ─────────────────────────────────────────────────────────────────────
_VERDICT_CODE = re.compile(r"(?<![A-Za-z_])(?:CONDITIONAL_PASS|INCONCLUSIVE|PASS|FAIL)(?![A-Za-z_])")
_VERDICT_CONTEXT_BEFORE = re.compile(r"판정\s*(?:은|이|결과는|결과|[:：])?\s*['\"‘’“”]?\s*$")
_SNAKE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+(?![A-Za-z0-9_])")


def name_exemptions(fs: FactSheet, client_inputs: dict | None) -> set[str]:
    """밑줄이 든 이름 가운데 문장에 써도 되는 것 — 클래스·지표·고객 입력의 글(모델명 'my_model_v2' 등)."""
    names: set[str] = set()

    def scan(obj: Any) -> None:
        if isinstance(obj, str):
            names.update(_SNAKE.findall(obj))
            names.update(_VERDICT_CODE.findall(obj))  # 클래스 라벨 'PASS'·모델명 'PASS-Net'(S5 검토 checks-9)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                scan(k)
                scan(v)
        elif isinstance(obj, list):
            for item in obj:
                scan(item)

    scan([m.display_name for m in fs.metrics])
    scan([p.label for p in fs.per_class])
    scan([r.label for r in fs.confusion_counts])
    if fs.confusion:
        scan(fs.confusion.labels)
    if fs.distribution:
        scan(list(fs.distribution.class_distribution))
    if fs.derived_prediction:
        scan(fs.derived_prediction.source_columns)
    scan([p.title for p in fs.problems])
    if fs.imbalance_effect:
        scan([fs.imbalance_effect.majority_class, fs.imbalance_effect.minority_class])
    scan(_fact_labels(fs))
    scan(client_inputs or {})
    return names


def _fact_labels(fs: FactSheet) -> list[str]:
    """facts 안의 클래스 이름 — 프론트가 표를 잘라도 facts 의 목록(편향·헷갈린 쌍·작은 표본·평균 비교 기준)에는 남아
    템플릿·AI 가 부를 수 있다(세션 A 검토 — 잘린 표 밖 'tiger_shark' 가 영어 코드로 걸렸다)."""
    facts = fs.facts
    if facts is None:
        return []
    labels = [r.label for r in facts.distribution_shift] + [r.label for r in facts.low_support_classes]
    if facts.weakest_class:
        labels.append(facts.weakest_class.label)
    if facts.prediction_bias:
        labels += [r.label for r in facts.prediction_bias.rows if r.label]
    for p in facts.top_confusion_pairs:
        labels += [p.actual, p.predicted]
    if facts.average_gap:
        labels.append(facts.average_gap.reference_class.label)
    return labels


# 판정 모순 검사 전에 가리는 이름 — 클래스 이름·고객이 준 클래스 설명에 판정 낱말이 들면('불합격품(기준 미달)') 이름을
# 부른 맞는 문장이 '미달했다'는 주장으로 읽힌다(세션 A 검토). 판정 낱말이 든 이름만 가린다.
_VERDICT_WORDS = re.compile(r"미달|불합격|미충족|합격|충족|판정|적합|FAIL|PASS|INCONCLUSIVE", re.IGNORECASE)


def verdict_masks(fs: FactSheet, client_inputs: dict | None) -> tuple[str, ...]:
    labels: list[str] = [p.label for p in fs.per_class] + [r.label for r in fs.confusion_counts]
    if fs.confusion:
        labels += list(fs.confusion.labels)
    if fs.distribution:
        labels += list(fs.distribution.class_distribution)
    if fs.imbalance_effect:
        labels += [fs.imbalance_effect.majority_class, fs.imbalance_effect.minority_class]
    labels += _fact_labels(fs)
    raw = (client_inputs or {}).get("class_label_descriptions") if isinstance(client_inputs, dict) else None
    descriptions = {str(k).strip(): str(v).strip() for k, v in raw.items() if v is not None} if isinstance(raw, dict) else {}
    found: set[str] = set()
    for label in labels:
        desc = descriptions.get(str(label).strip())
        if desc:
            found.add(f"{label}({desc})")
    found.update(t for t in descriptions.values() if t)
    found.update(str(x) for x in labels)
    return tuple(sorted((t for t in found if t and _VERDICT_WORDS.search(t)), key=len, reverse=True))


def mask_names(texts: list[str], masks: tuple[str, ...]) -> list[str]:
    if not masks:
        return texts
    out = []
    for text in texts:
        if isinstance(text, str):
            for name in masks:
                text = text.replace(name, "□")
        out.append(text)
    return out


def code_words(texts: list[str], exempt: set[str]) -> list[str]:
    found: list[str] = []
    for text in texts:
        if not isinstance(text, str):
            continue
        for m in _VERDICT_CODE.finditer(text):
            # 이름과 겹치는 판정 코드는 '판정' 문맥("판정은 FAIL")일 때만 잡는다 — 그 밖은 클래스·모델 이름이다.
            if m.group(0) not in exempt or _VERDICT_CONTEXT_BEFORE.search(text[:m.start()]):
                found.append(m.group(0))
        found.extend(w for w in _SNAKE.findall(text) if w not in exempt)
    return sorted(set(found))


def _cites_allowed_number(text: str, whitelist: set) -> bool:
    stripped = _CTX_TOKEN_RE.sub(" ", _normalize_minus(text or ""))
    return any(_canon(tok) in whitelist for tok in _NUMBER_RE.findall(stripped))


def check_section(
    section_id: str,
    parts: dict,
    *,
    whitelist: set,
    verdict: str,
    conclusions: list[Conclusion],
    problem_ids: list[str],
    exempt: set[str],
    masks: tuple[str, ...] = (),
) -> SectionCheck:
    """AI 가 쓴 칸 하나를 검사한다. parts 는 그 칸의 응답 필드(점 표기) → 값(9칸의 권고는 dict 목록)."""
    result = SectionCheck(section=section_id)
    texts = [t for t in section_texts(section_id, parts) if isinstance(t, str)]

    # ⑤ 빈 칸 — 7.3 은 비면 '해당 없음'이라 빈 글이 정상이다. 빈 칸이면 다른 검사는 의미가 없다.
    if section_id in REQUIRED_NON_EMPTY:
        main = texts[:2] if section_id == "9" else texts
        if any(not t.strip() for t in main) or (section_id == "9" and not parts.get("recommendations")):
            result.format.append("empty")
            return result
    elif not any(t.strip() for t in texts):
        return result  # 7.3 빈 글 — 통과(고정 문구 15)

    result.numbers = verify_grounding(texts, whitelist).violations
    result.contradiction = bool(find_verdict_contradictions(mask_names(texts, masks), verdict))
    result.code_words = code_words(texts, exempt)
    if result.code_words:
        result.format.append("code_words")
    if section_id == "7.3" and not _cites_allowed_number(texts[0], whitelist):
        result.format.append("no_citation")

    if section_id == "9":
        recs = parts.get("recommendations") or []
        for rec in recs:
            result.impact_numbers.extend(predicted_numbers(rec.get("expected_impact", "")))
        answered = {str(r.get("problem_id") or "").strip() for r in recs}
        result.missing_recommendations = [pid for pid in problem_ids if pid not in answered]

    for c in conclusions:
        if c.section == section_id and c.d24_values and not d24_present(" ".join(texts), c.d24_values):
            result.d24_missing = True
    return result
