"""심판 라운드 실호출 실행기(scripts/judge_live.py) 초안 시험 — 세션 1 J0.

실호출은 하지 않는다. `--live` 없이 클라이언트를 만들지 않는지, 장부·상한·누계 알림·dev 읽기 경계를 본다.
"""
import csv
import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "judge_live", Path(__file__).resolve().parents[1] / "scripts" / "judge_live.py"
)
jl = importlib.util.module_from_spec(_SPEC)
sys.modules["judge_live"] = jl  # dataclass 가 모듈을 찾는다
_SPEC.loader.exec_module(jl)


def _runner(tmp_path, *, usd=4.0, runs=40, answers=(), mode="live"):
    out: list[str] = []
    asked: list[str] = []
    replies = iter(answers)

    def confirm(prompt):
        asked.append(prompt)
        return next(replies)

    runner = jl.Runner(jl.Ledger(tmp_path / "장부.csv"), jl.Budget(usd, runs), mode, confirm=confirm, out=out.append)
    return runner, out, asked


# ── 장부 ──


def test_new_ledger_has_exact_header(tmp_path):
    jl.Ledger(tmp_path / "장부.csv")
    with (tmp_path / "장부.csv").open(encoding="utf-8") as f:
        assert next(csv.reader(f)) == list(jl.LEDGER_COLUMNS)


def test_ledger_columns_match_validation_folder_list():
    # 저장소 밖 검증 폴더의 열 목록과 같은 목록이다(옆 폴더가 없으면 건너뜀 — CI 는 저장소 하나만 받는다).
    listing = Path(__file__).resolve().parents[2] / "JUDGE_ROUND" / "검증" / "장부_열.txt"
    if not listing.exists():
        pytest.skip("JUDGE_ROUND 폴더 없음")
    assert tuple(listing.read_text(encoding="utf-8").split()) == jl.LEDGER_COLUMNS


def test_ledger_with_other_header_is_refused(tmp_path):
    path = tmp_path / "장부.csv"
    path.write_text("run_id,at\n", encoding="utf-8")
    with pytest.raises(jl.RunnerError, match="머리줄"):
        jl.Ledger(path)


def test_append_fills_id_and_time_and_rejects_unknown_columns(tmp_path):
    ledger = jl.Ledger(tmp_path / "장부.csv")
    first = ledger.append({"stage": "smoke", "cost_usd": "0.10"})
    second = ledger.append({"stage": "smoke"})
    assert (first["run_id"], second["run_id"]) == ("R-0001", "R-0002")
    assert first["at"].endswith("Z")
    assert ledger.totals() == (pytest.approx(0.10), 2, 1)
    with pytest.raises(jl.RunnerError, match="없는 열"):
        ledger.append({"stage": "smoke", "비용": "1"})


# ── 시작 전 상한·확인 ──


def test_start_prints_estimate_and_totals_first(tmp_path):
    runner, out, _ = _runner(tmp_path)
    runner.ledger.append({"stage": "smoke", "cost_usd": "0.30"})
    runner.start(jl.BatchPlan("seal-generate", 6, 1.2), yes=True)
    assert out == ["[live] 묶음 seal-generate: 예상 6회·$1.20 | 누계 $0.30·1회(비용 모름 0회) | 상한 $4.00·40회"]


def test_start_without_yes_stops(tmp_path):
    runner, out, _ = _runner(tmp_path)
    with pytest.raises(jl.RunnerError, match="--yes"):
        runner.start(jl.BatchPlan("smoke", 3, 0.3), yes=False)
    assert len(out) == 1  # 예상은 찍고 멈춘다


def test_start_refuses_batch_over_cost_cap(tmp_path):
    runner, _, _ = _runner(tmp_path, usd=4.0)
    runner.ledger.append({"stage": "seal", "cost_usd": "3.50"})
    runner.start(jl.BatchPlan("ok", 1, 0.50), yes=True)  # 딱 상한까지는 된다
    with pytest.raises(jl.RunnerError, match="비용 상한"):
        runner.start(jl.BatchPlan("pilot", 1, 0.51), yes=True)


def test_start_refuses_batch_over_run_cap_even_when_cost_unknown(tmp_path):
    runner, _, _ = _runner(tmp_path, runs=5)
    for _ in range(3):
        runner.ledger.append({"stage": "smoke"})
    runner.start(jl.BatchPlan("ok", 2, None), yes=True)
    with pytest.raises(jl.RunnerError, match="회차 상한"):
        runner.start(jl.BatchPlan("pilot", 3, None), yes=True)


def test_budget_from_env(monkeypatch):
    monkeypatch.delenv("JUDGE_LIVE_BUDGET_USD", raising=False)
    monkeypatch.delenv("JUDGE_LIVE_MAX_RUNS", raising=False)
    assert jl.Budget.from_env() == jl.Budget(4.0, 40)
    monkeypatch.setenv("JUDGE_LIVE_BUDGET_USD", "24.5")
    monkeypatch.setenv("JUDGE_LIVE_MAX_RUNS", "130")
    assert jl.Budget.from_env() == jl.Budget(24.5, 130)


# ── 누계 알림 ──


@pytest.mark.parametrize(
    "before, after, crossed",
    [(0.0, 9.99, False), (9.99, 10.0, True), (9.0, 10.5, True), (10.0, 19.9, False), (19.0, 31.0, True)],
)
def test_crossed_checkpoint(before, after, crossed):
    assert jl.crossed_checkpoint(before, after) is crossed


def test_record_stops_at_ten_dollars_unless_confirmed(tmp_path):
    runner, _, asked = _runner(tmp_path, usd=100, answers=["계속", "아니"])
    runner.record({"stage": "repro", "cost_usd": "9.50"})
    assert asked == []
    runner.record({"stage": "repro", "cost_usd": "0.60"})  # 10.10 — 묻고 계속
    assert len(asked) == 1 and "$10" in asked[0]
    runner.record({"stage": "repro", "cost_usd": "9.00"})  # 19.10 — 안 묻는다
    assert len(asked) == 1
    with pytest.raises(jl.RunnerError, match="멈춘다"):
        runner.record({"stage": "repro", "cost_usd": "1.00"})  # 20.10 — 거절
    assert runner.ledger.totals()[1] == 4  # 멈춰도 그 실행은 장부에 남는다


def test_record_stops_every_twenty_unknown_cost_runs(tmp_path):
    runner, _, asked = _runner(tmp_path, answers=["계속"])
    for _ in range(19):
        runner.record({"stage": "smoke"})
    assert asked == []
    runner.record({"stage": "smoke"})
    assert len(asked) == 1 and "20회" in asked[0]


def test_record_marks_mode(tmp_path):
    runner, _, _ = _runner(tmp_path, mode="dev")
    assert runner.record({"stage": "dev"})["mode"] == "dev"
    with pytest.raises(jl.RunnerError, match="모드"):
        jl.Runner(runner.ledger, jl.Budget(1, 1), "verify")  # verify 는 J8 에 붙는다


# ── dev 읽기 경계 ──


def test_dev_mode_cannot_read_validation_folder(tmp_path, monkeypatch):
    vdir = tmp_path / "검증"
    (vdir / "00_봉인").mkdir(parents=True)
    monkeypatch.setenv("JUDGE_VALIDATION_DIR", str(vdir))
    with pytest.raises(jl.RunnerError, match="검증 폴더"):
        jl.guard_read(vdir / "00_봉인" / "x.json", "dev")
    with pytest.raises(jl.RunnerError, match="검증 폴더"):
        jl.guard_read(vdir, "dev")
    with pytest.raises(jl.RunnerError, match="검증 폴더"):
        jl.guard_read(tmp_path / "다른" / ".." / "검증" / "00_봉인", "dev")  # 우회 경로도
    assert jl.guard_read(tmp_path / "judge_dev" / "a.json", "dev") == (tmp_path / "judge_dev" / "a.json").resolve()
    assert jl.guard_read(vdir / "00_봉인" / "x.json", "live")  # live 는 봉인 생성에 쓴다


def test_dev_mode_needs_to_know_validation_folder(tmp_path, monkeypatch):
    monkeypatch.delenv("JUDGE_VALIDATION_DIR", raising=False)
    with pytest.raises(jl.RunnerError, match="JUDGE_VALIDATION_DIR"):
        jl.guard_read(tmp_path / "a.json", "dev")


# ── --live 없이는 클라이언트를 만들지 않는다 ──


def test_no_live_builds_no_client_even_with_keys(monkeypatch):
    import anthropic
    import openai

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def boom(*a, **k):
        raise AssertionError("--live 없이 진짜 클라이언트를 만들었다")

    monkeypatch.setattr(anthropic, "AsyncAnthropic", boom)
    monkeypatch.setattr(openai, "AsyncOpenAI", boom)
    clients = jl.make_clients(live=False)
    assert (clients.anthropic, clients.openai) == (None, None)


def test_live_builds_clients_without_sdk_retries(monkeypatch):
    import anthropic
    import openai
    import dotenv

    made = {}
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda **k: made.setdefault("anthropic", k))
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **k: made.setdefault("openai", k))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    jl.make_clients(live=True)
    assert made["anthropic"]["max_retries"] == 0 and made["openai"]["max_retries"] == 0


def test_live_without_any_key_stops(monkeypatch):
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(jl.RunnerError, match="둘 다 없다"):
        jl.make_clients(live=True)


# ── 명령줄 ──


def test_status_command(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("JUDGE_LIVE_BUDGET_USD", raising=False)
    monkeypatch.delenv("JUDGE_LIVE_MAX_RUNS", raising=False)
    path = tmp_path / "장부.csv"
    jl.Ledger(path).append({"stage": "smoke", "cost_usd": "0.25"})
    assert jl.main(["status", "--ledger", str(path)]) == 0
    assert capsys.readouterr().out == "누계 $0.25·1회(비용 모름 0회) | 상한 $4.00·40회\n"
