"""심판 라운드 실호출 실행기(scripts/judge_live.py) 초안 시험 — 세션 1 J0.

실호출은 하지 않는다. `--live` 없이 클라이언트를 만들지 않는지, 장부·상한·누계 알림·dev 읽기 경계를 본다.
"""
import csv
import json
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


# ── smoke(J1) — 가짜 OpenAI 로 흐름만 ──


async def test_smoke_flow_with_fake_client(tmp_path):
    from tests import judge_fakes as F

    fake = F.FakeOpenAI([
        F.response([F.reasoning_item("enc-1"), F.function_call_item("calc", {"expr": "38/(38+11)"}, "call_1")]),
        F.json_response({"ratio": 0.7755}),
        F.json_response({"claims": [{"quote": "0.912", "kind": "number"}]}, model="gpt-6-luna"),
        F.json_response({"claims": []}, model="gpt-6-luna", use=F.usage(input_tokens=2000, cached=1536)),
    ])
    runner, out, _ = _runner(tmp_path)
    summary = await jl.smoke(runner, jl.Clients(openai=fake), {"criteria_version": "1.0-dev"})
    tl = summary["tool_loop"]
    assert (tl["turns"], tl["tool_calls"], tl["encrypted_reasoning_items"], tl["final"]) == (2, 1, 2, {"ratio": 0.7755})  # 두 응답 모두 암호화 추론을 실음
    # 두 번째 턴 입력에 암호화 추론·함수 호출·도구 결과가 되돌아간다(응답 보관 끔)
    second_input = fake.calls[1]["input"]
    assert [i.get("type") for i in second_input[1:]] == ["reasoning", "function_call", "function_call_output"]
    assert json.loads(second_input[-1]["output"]) == {"value": pytest.approx(38 / 49)}
    assert [s["cached_input_tokens"] for s in summary["luna"]] == [0, 1536]
    rows = runner.ledger.rows()
    assert [(r["stage"], r["report"], r["repeat"]) for r in rows] == [
        ("smoke", "tool_loop", "1"), ("smoke", "tool_loop", "2"), ("smoke", "luna_extract", "1"), ("smoke", "luna_extract", "2")]
    assert rows[0]["criteria_version"] == "1.0-dev" and rows[2]["model_served"] == "gpt-6-luna" and rows[0]["cost_usd"]


def test_smoke_command_needs_live_and_yes(tmp_path):
    path = tmp_path / "장부.csv"
    assert jl.main(["smoke", "--ledger", str(path)]) == 2  # --yes 없음
    assert jl.main(["smoke", "--ledger", str(path), "--yes"]) == 2  # --live 없음
    assert jl.Ledger(path).totals()[1] == 0


# ── seal-generate(J2 끝) — 가짜 Claude(거절 → 규칙 문장)로 흐름만 ──


class _RefusingClaude:
    def __init__(self):
        from types import SimpleNamespace

        self.calls = 0
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def with_options(self, **_kwargs):
        return self

    async def _create(self, **_kwargs):
        from types import SimpleNamespace

        self.calls += 1
        return SimpleNamespace(model="fake-claude", stop_reason="refusal", stop_details=None, content=[],
                               usage=SimpleNamespace(input_tokens=0, output_tokens=0, iterations=None))


def test_seal_generate_writes_read_only_sealed_files(tmp_path):
    import os
    import shutil

    src = Path(__file__).parent / "data" / "judge_dev"
    req_dir = tmp_path / "요청"
    req_dir.mkdir()
    for name in ("dev_binary.request.json", "dev_multilabel.request.json"):
        shutil.copy(src / name, req_dir / name)
    out = tmp_path / "봉인"
    runner, _, _ = _runner(tmp_path)
    fake = _RefusingClaude()
    summaries = jl.seal_generate(runner, req_dir, out, {"criteria_version": "1.0-dev"}, anthropic_client=fake)
    assert fake.calls > 0
    assert [s["tag"] for s in summaries] == ["dev_binary", "dev_multilabel"]
    assert all(s["status"] == "done" and set(s["sources"]) == {"7.1", "7.2", "7.3", "8.1", "8.2", "8.3", "9"} for s in summaries)
    # 요약에는 서술 글이 없다(화면에 찍지 않음)
    assert "interpretation" not in json.dumps(summaries, ensure_ascii=False)
    sealed = json.loads((out / "dev_binary.json").read_text(encoding="utf-8"))
    assert set(sealed) == {"tag", "request", "created", "view", "internal"}
    assert sealed["request"] == json.loads((req_dir / "dev_binary.request.json").read_text(encoding="utf-8"))
    sums = (out / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert [line.split("  ")[1] for line in sums] == ["dev_binary.json", "dev_multilabel.json"]
    assert (out / "봉인시각.txt").exists()
    assert not os.access(out / "dev_binary.json", os.W_OK)
    rows = runner.ledger.rows()
    assert [(r["stage"], r["report"]) for r in rows] == [("seal_generate", "dev_binary"), ("seal_generate", "dev_multilabel")]
    assert rows[0]["result_sha256"] == summaries[0]["sha256"]
    # 한 번 봉인한 폴더는 다시 만들지 않는다
    with pytest.raises(jl.RunnerError, match="이미 봉인"):
        jl.seal_generate(runner, req_dir, out, {}, anthropic_client=fake)


def test_seal_generate_refuses_without_claude_client(tmp_path):
    req_dir = tmp_path / "요청"
    req_dir.mkdir()
    (req_dir / "x.request.json").write_text(
        (Path(__file__).parent / "data" / "judge_dev" / "dev_binary.request.json").read_text(encoding="utf-8"), encoding="utf-8")
    runner, _, _ = _runner(tmp_path)
    with pytest.raises(jl.RunnerError, match="서술 클라이언트가 없다"):
        jl.seal_generate(runner, req_dir, tmp_path / "봉인", {})
    assert runner.ledger.rows() == []


# ── pilot(J4 끝) — 가짜 OpenAI 로 흐름만 ──


async def test_pilot_flow_with_fake_judge(tmp_path):
    from tests import judge_fakes as F
    from tests.test_judge_core import _empty_answers, router

    fake = F.FakeOpenAI([router(_empty_answers(None))] * 30)
    runner, _, _ = _runner(tmp_path)
    path = Path(__file__).parent / "data" / "judge_dev" / "dev_multiclass.request.json"
    out = tmp_path / "pilot"
    summaries = await jl.pilot(runner, jl.Clients(openai=fake), [path], out, "template", {"criteria_version": "1.0-dev"})
    assert summaries[0]["failure"] == "no_claims"  # 가짜가 주장 0개라 보류 — 흐름만 본다
    saved = json.loads((out / "dev_multiclass.template.json").read_text(encoding="utf-8"))
    assert saved["record"]["material_problem"] is None and saved["result"]["conclusion"]["narrative"]
    rows = runner.ledger.rows()
    assert [(r["stage"], r["report"], r["error"]) for r in rows] == [("pilot_judge", "dev_multiclass", "no_claims")]


def test_pilot_refuses_validation_folder(tmp_path, monkeypatch):
    vdir = tmp_path / "검증"
    (vdir / "00_봉인").mkdir(parents=True)
    sealed = vdir / "00_봉인" / "bc_strong.request.json"
    sealed.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("JUDGE_VALIDATION_DIR", str(vdir))
    assert jl.main(["pilot", "--ledger", str(tmp_path / "l.csv"), "--request", str(sealed), "--out", str(tmp_path / "o"),
                    "--live", "--yes"]) == 2
