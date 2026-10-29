"""app/judge/validation/runs.py — 검증 실행(V2)의 회차 계획·이어 하기·순서 섞기(계획 17-3, 설계 7.2). 순수 함수.

묶음 넷과 기본 회차는 줄인 V2 안(민서 결정 2026-10-29: 재현성 3건 × 3회 = 9, 결함 6개 × 1회 = 6, sham 3 → 채점 18회).
정확한 회차는 V1 끝에 확정한다 — 실행기 인자로 바꿀 수 있다. 추출(재현율·정밀도) 묶음은 채점을 따로 부르지 않는다:
재현성 회차의 채점 기록에 전처리(주장 추출) 결과가 들어 있어 그것을 정답지와 짝짓는다(같은 3건 — Q9 답).

회차 하나 = RunKey(묶음, 성적서, 변형, 회차 번호). 장부 한 줄과 1:1 이다(stage = "validate_<묶음>", report, variant, repeat).
이어 하기: 같은 모드·같은 기준 지문으로 결과 파일까지 적히고 **게이트 판정이 난**(pass·blocked) 장부 줄이 있으면 끝난 회차로
보고 건너뛴다. 채점이 실패한 회차(judge_once 가 gate=None·failure 로 끝남 — 결과 파일·장부 줄은 써진다)는 끝나지 않은 것이라
다시 돈다(실패 줄의 비용은 장부에 그대로 남아 누계에 든다). 기준 지문이 다르면(버전을 올린 재검증) 끝나지 않은 것으로 본다.

상호작용
- 의존(import): random, dataclasses
- 사용처: scripts/judge_validate.py
"""
from __future__ import annotations

import random
from dataclasses import dataclass

BATCHES = ("reproducibility", "defects", "sham", "extraction")
STAGE_PREFIX = "validate_"
BASE_VARIANT = "base"
# 판정이 난 회차의 게이트 값(app.judge.core.JudgeRun.gate). 빈칸·그 밖의 값(실패 회차)은 끝나지 않은 것.
DONE_GATES = ("pass", "blocked")

# 채점 1회 비용(계획 §마-2 J4 끝 실측 — AI 서술 채점 $0.15). 묶음 예상 비용 = 회차 × 이 값.
COST_PER_SCORING_USD = 0.15
# 묶음 예산 = 예상 × 이 배수(실측 흔들림 여유). 넘으면 그 회차 뒤에 멈춘다(이어 하기로 계속).
BATCH_BUDGET_FACTOR = 1.5


@dataclass(frozen=True)
class BatchDefault:
    items: int  # 재현성: 성적서 수, 결함·sham: 변형 수
    repeats: int


DEFAULTS: dict[str, BatchDefault] = {
    "reproducibility": BatchDefault(items=3, repeats=3),
    "defects": BatchDefault(items=6, repeats=1),
    "sham": BatchDefault(items=3, repeats=1),
    "extraction": BatchDefault(items=0, repeats=0),  # 채점 없음 — 재현성 기록을 쓴다
}


@dataclass(frozen=True, order=True)
class RunKey:
    batch: str
    report: str
    variant: str
    repeat: int

    @property
    def stage(self) -> str:
        return STAGE_PREFIX + self.batch


def plan_runs(batch: str, items: list[tuple[str, str]], repeats: int) -> list[RunKey]:
    """(성적서, 변형) 목록 × 회차 → 회차 목록(정렬). 재현성의 변형은 늘 base."""
    if batch not in BATCHES:
        raise ValueError(f"모르는 묶음: {batch}")
    if batch == "extraction":
        return []
    if repeats < 1:
        raise ValueError("회차는 1 이상")
    if len(set(items)) != len(items):
        raise ValueError("같은 (성적서, 변형)이 두 번 있다")
    return sorted(RunKey(batch, report, variant, r) for report, variant in items for r in range(1, repeats + 1))


def key_of_row(row: dict) -> RunKey | None:
    stage = row.get("stage", "")
    if not stage.startswith(STAGE_PREFIX) or not row.get("repeat", "").isdigit():
        return None
    return RunKey(stage[len(STAGE_PREFIX):], row.get("report", ""), row.get("variant", ""), int(row["repeat"]))


def finished_row(row: dict, *, fingerprint: str, mode: str) -> RunKey | None:
    """이 장부 줄이 끝난 회차면 그 RunKey, 아니면 None — 같은 모드·같은 기준 지문, 결과 파일이 적혔고 게이트 판정이 난 줄.
    예외로 끝나 결과 파일이 없는 줄·채점이 실패한 줄(gate 빈칸)은 끝나지 않은 것. 이어 하기와 report 가 같은 규칙을 쓴다."""
    key = key_of_row(row)
    if (key and row.get("mode") == mode and row.get("manifest_sha256") == fingerprint
            and row.get("result_file", "").strip() and row.get("gate", "") in DONE_GATES):
        return key
    return None


def done_keys(rows: list[dict], *, fingerprint: str, mode: str) -> set[RunKey]:
    """끝난 회차(finished_row 규칙)."""
    return {k for row in rows if (k := finished_row(row, fingerprint=fingerprint, mode=mode)) is not None}


def pending(keys: list[RunKey], rows: list[dict], *, fingerprint: str, mode: str) -> list[RunKey]:
    done = done_keys(rows, fingerprint=fingerprint, mode=mode)
    return [k for k in keys if k not in done]


def shuffled(keys: list[RunKey], seed: int) -> list[RunKey]:
    """실행 순서 섞기(시드 기록) — 같은 시드면 같은 순서. 회차 번호가 시간 순서와 엮이지 않게(캐시·시간대 흔들림)."""
    order = sorted(keys)
    random.Random(seed).shuffle(order)
    return order


def estimate_usd(runs: int) -> float:
    return round(runs * COST_PER_SCORING_USD, 2)


def batch_budget_usd(runs: int) -> float:
    return round(runs * COST_PER_SCORING_USD * BATCH_BUDGET_FACTOR, 2)
