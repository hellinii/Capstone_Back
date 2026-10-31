"""tests/test_judge_criteria_manifest.py — 심판 기준 묶음의 지문(설계 7.4·7.5, 계획 17-1).

- 파일 ↔ MANIFEST: 기준 파일이 MANIFEST 를 다시 쓰지 않고 바뀌면 빨개진다.
- 버전 기록: VERSION 이 고정 버전(이름에 -dev 가 없음)이면 VERSIONS.json 의 그 버전 지문 == 지금 지문. 고정 뒤 MANIFEST 를
  다시 써서 초록으로 만들려 해도 기록과 달라 빨개진다 → 버전을 올리고 다시 검증해야 한다.
- 기록은 덧붙이기만: 아래 KNOWN 에 적은 줄이 VERSIONS.json 에 글자 그대로 남아 있어야 한다(지우거나 바꾸면 빨개짐).
  새 버전을 고정할 때 VERSIONS.json 과 KNOWN 에 같은 줄을 더한다.
- CHANGELOG 에 지금 VERSION 절이 있다. model.json 이 코드 기본값(app/judge/config)과 같다.
"""
from __future__ import annotations

import re

import pytest

from app.judge import config, criteria

# 고정한 버전 → 기준 지문. 덧붙이기만 한다(V1 에서 "1.0" 을 처음 적는다 — scripts/judge_criteria_manifest.py freeze 가 찍는 줄).
# 왜 시험에 사본을 박나(git 이력·태그 대신): CI 는 저장소를 얕게 받고 태그를 받지 않아 이력을 읽는 시험은 CI 에서 깨진다.
# 사본이 시험 파일에 있으면 VERSIONS.json 의 기록 줄을 지우거나 바꾸려면 두 파일을 함께 고쳐야 하고, 그 diff 가 검토에 드러난다.
KNOWN: dict[str, str] = {
    "1.0": "2cbc32785d6deec818f417a193711d7242a6defcab86c17123c4d348d9e18e58",
    "1.1": "25f70440c5851b80e8950c5e2c1dd573ad35dfb47807aa43cc51737a325c8d6c",
    "1.2": "613a6f39883ee662be0b8af0ee3e4fa0a63d56fea0996c92ea35301f99349c18",
    "1.3": "e25d2af4f893846f7b9c75e91a5ec2cf99d082e76fbe6ef0b9e3d75e513eac3f",
    "1.4": "4c6017d1fb711338c51ff6c4827976910de1d646783f9405279da2ab83f7728f",
    "1.5": "4a8bff76cc5d0a8d5fdec8ee69ea0a4524fde7301d330502c95320903e78aa88",
    "1.6": "cc68b6ed2d27f63bc476e42a27cfcc86630397ffde2ab8f1974f401c6210b1bf",
    "1.7": "1093a56d30a14fb4706aef939d4fbab05c492e49c585b76309b65ac6a4f50fd4",
    "1.8": "8aea98bca2e1350f83384dd11481bfb66d0e726f0adfa28a819dea3b1a503f9e",
    "1.9": "5369df5b1b6d4455ac8c5b766439a4851e95f88d2a7d571fa5f64498796f0e39",
    "1.10": "15ad1bb187fe367a362bd34ff1f8d15cc4195669075d5dbb2a7615cc7e016dff",
    "1.11": "95f47dfa273ddad3e60e4b014f3da288466caf569a5668f8a441074b166c4988",
    "1.12": "7e95790ea2eaea68829b86d1fa8cfc885a714d7fb139a9f5d9bf7f5f1191e205",
    "1.13": "742c21ee75ae8e31a0f85b5ea35fa2e051abe39b6bfe0f06ccc984cf6daf5ca1",
    "1.14": "37dc2e231e60b806da8bb40db624ef3af0f8be3fb5872794a23c93a8d81d4038",
    "1.15": "cd037e9bac9cae64f985acbecd396b5c50dd3d5fc2e172a7bf6f711d291e7c1f",
}


def append_only_problems(known: dict[str, str], recorded: dict[str, str]) -> list[str]:
    """기록(VERSIONS.json)이 사본(KNOWN)을 앞에서부터 글자 그대로 담고 있나 — 지움·바꿈·끼워 넣기를 사유로."""
    problems = []
    for ver, fp in known.items():
        if ver not in recorded:
            problems.append(f"기록 줄이 지워졌다: {ver}")
        elif recorded[ver] != fp:
            problems.append(f"기록 줄이 바뀌었다: {ver}")
    if not problems and list(recorded)[: len(known)] != list(known):
        problems.append("기록 순서가 바뀌었다(새 줄은 끝에만 덧붙인다)")
    return problems


def test_manifest_matches_files():
    files = criteria.bundle_files()
    assert len(files) >= 3  # VERSION·model.json·로더는 늘 있다
    assert criteria.recorded_manifest() == criteria.manifest_text(), (
        "기준 파일이 바뀌었는데 MANIFEST 가 그대로다 — 이유를 CHANGELOG 에 적고 scripts/judge_criteria_manifest.py 로 다시 쓴다"
        " (고정한 버전이면 VERSION 을 올리고 다시 검증)"
    )


def test_manifest_excludes_records_and_lists_relative_paths():
    names = [line.split("  ", 1)[1] for line in criteria.recorded_manifest().splitlines()]
    assert names == sorted(names)
    assert not set(names) & criteria.NOT_IN_MANIFEST
    assert "VERSION" in names and "model.json" in names


def test_fixed_version_fingerprint_is_recorded():
    assert criteria.version_problem() is None


def test_version_problem_cases():
    fp = "a" * 64
    assert criteria.version_problem("1.0-dev", {}, fp) is None
    assert criteria.version_problem("1.0-dev", {"1.0-dev": fp}, fp) is not None
    assert criteria.version_problem("1.0", {"1.0": fp}, fp) is None
    assert criteria.version_problem("1.0", {}, fp) is not None
    assert "VERSION 을 올리고" in criteria.version_problem("1.0", {"1.0": "b" * 64}, fp)


def test_version_records_are_append_only():
    recorded = criteria.recorded_versions()
    assert append_only_problems(KNOWN, recorded) == []
    assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in recorded.values())
    # 기록에만 있고 사본에 없는 버전 = 시험 사본을 덧붙이지 않은 고정(또는 몰래 더한 줄)
    assert set(recorded) == set(KNOWN), "VERSIONS.json 에 새 기록이 있으면 같은 줄을 이 시험의 KNOWN 에 덧붙인다"


def test_append_only_detects_delete_change_and_reorder():
    # 이빨: 사본과 기록이 어긋나는 세 꼴을 각각 잡는다.
    known = {"1.0": "a" * 64, "1.1": "b" * 64}
    assert append_only_problems(known, dict(known)) == []
    assert append_only_problems(known, {**known, "1.2": "c" * 64}) == []  # 끝에 덧붙이기는 된다
    assert append_only_problems(known, {"1.1": "b" * 64}) == ["기록 줄이 지워졌다: 1.0"]
    assert append_only_problems(known, {"1.0": "d" * 64, "1.1": "b" * 64}) == ["기록 줄이 바뀌었다: 1.0"]
    assert append_only_problems(known, {"1.1": "b" * 64, "1.0": "a" * 64}) == ["기록 순서가 바뀌었다(새 줄은 끝에만 덧붙인다)"]


def test_changelog_has_current_version():
    assert f"## {criteria.version()}\n" in (criteria.HERE / "CHANGELOG.md").read_text(encoding="utf-8")


def test_changelog_has_every_recorded_version():
    text = (criteria.HERE / "CHANGELOG.md").read_text(encoding="utf-8")
    assert [v for v in criteria.recorded_versions() if f"## {v}\n" not in text] == []


def test_model_settings_match_code_defaults():
    m = criteria.load_json("model.json")
    assert (m["judge_model"], m["extract_model"], m["effort"], m["extract_effort"]) == (
        config.DEFAULT_JUDGE_MODEL, config.DEFAULT_EXTRACT_MODEL, config.DEFAULT_EFFORT, config.DEFAULT_EXTRACT_EFFORT
    )


def test_fingerprint_changes_when_a_file_changes(tmp_path, monkeypatch):
    # 이빨: 파일 한 바이트가 바뀌면 MANIFEST 글과 지문이 달라진다.
    for p in criteria.bundle_files():
        dst = tmp_path / p.relative_to(criteria.HERE)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(p.read_bytes())
    monkeypatch.setattr(criteria, "HERE", tmp_path)
    before = criteria.manifest_text()
    (tmp_path / "model.json").write_text((tmp_path / "model.json").read_text() + " ", encoding="utf-8")
    after = criteria.manifest_text()
    assert before != after and criteria.fingerprint(before) != criteria.fingerprint(after)


# 주장 규칙 잠금(심판 라운드 J4 끝, 계획 Q9) — 사람 정답지 표시가 이 규칙으로 시작한다. 바꾸면 표시를 다시 해야 한다.
CLAIMS_LOCK = "2fd85afecccda1bea6a3d3270c55c1068e06ab589915b7457e5d546cb6403a65"


def test_claim_rules_are_locked_for_annotation():
    import hashlib

    got = hashlib.sha256((criteria.HERE / "claims.json").read_bytes()).hexdigest()
    assert got == CLAIMS_LOCK, (
        "주장 규칙(claims.json)이 잠근 뒤 바뀌었다 — 사람 정답지 표시(검증/01_정답지)를 새 규칙으로 다시 해야 한다. "
        "바꾼 이유를 CHANGELOG 에 적고 표시자에게 알린 뒤 이 값을 고친다"
    )


# ── V1 고정 명령(scripts/judge_criteria_manifest.py freeze) — 임시 사본 폴더에서만 돈다 ──


def _manifest_tool():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "judge_criteria_manifest", Path(__file__).resolve().parents[1] / "scripts" / "judge_criteria_manifest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bundle_copy(tmp_path):
    """고정 전(1.0-dev, 기록 없음, MANIFEST 맞음) 사본 — 저장소가 이미 고정됐어도(1.0) 같은 출발점에서 freeze 를 본다."""
    for p in [*criteria.bundle_files(), criteria.HERE / "MANIFEST.sha256", criteria.HERE / "VERSIONS.json",
              criteria.HERE / "CHANGELOG.md"]:
        dst = tmp_path / p.relative_to(criteria.HERE)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(p.read_bytes())
    (tmp_path / "VERSION").write_text("1.0-dev\n", encoding="utf-8")
    (tmp_path / "VERSIONS.json").write_text("{}\n", encoding="utf-8")
    tool = _manifest_tool()
    tool.write_manifest(tmp_path)
    with (tmp_path / "CHANGELOG.md").open("a", encoding="utf-8") as f:
        f.write("\n## 1.0\n\n- 시험용 고정.\n")
    with tool.bundle_at(tmp_path):  # 출발점 확인 — 고정 전·기록 없음·지문 맞음
        assert (criteria.version(), criteria.recorded_versions()) == ("1.0-dev", {})
        assert criteria.recorded_manifest() == criteria.manifest_text()
    return tmp_path


def test_freeze_writes_version_manifest_and_one_record(tmp_path, monkeypatch):
    tool = _manifest_tool()
    here = _bundle_copy(tmp_path)
    fp = tool.freeze("1.0", here)
    monkeypatch.setattr(criteria, "HERE", here)
    assert criteria.version() == "1.0"
    assert criteria.recorded_manifest() == criteria.manifest_text()
    assert criteria.recorded_versions() == {"1.0": fp} and fp == criteria.fingerprint()
    assert criteria.version_problem() is None
    # 고정 뒤 기준 파일을 바꾸고 MANIFEST 만 다시 써도 기록과 달라 빨개진다(7.4 — 버전을 올려야 한다).
    (here / "model.json").write_text((here / "model.json").read_text(encoding="utf-8") + " ", encoding="utf-8")
    tool.write_manifest(here)
    assert "VERSION 을 올리고" in criteria.version_problem()


def test_freeze_main_prints_known_line_and_tag(tmp_path, capsys):
    tool = _manifest_tool()
    here = _bundle_copy(tmp_path)
    assert tool.main(["freeze", "1.0", "--criteria-dir", str(here)]) == 0
    out = capsys.readouterr().out
    assert '"1.0": "' in out and "git tag -a judge-criteria-v1.0" in out


@pytest.mark.parametrize("case, msg", [
    ("dev_name", "숫자.숫자"),
    ("already", "이미 고정 기록"),
    ("dirty", "MANIFEST 가 다르다"),
    ("no_changelog", "CHANGELOG.md 에 '## 1.0'"),
])
def test_freeze_refuses_and_changes_nothing(tmp_path, case, msg):
    tool = _manifest_tool()
    here = _bundle_copy(tmp_path)
    version = "1.0"
    if case == "dev_name":
        version = "1.0-dev"
    elif case == "already":
        (here / "VERSIONS.json").write_text('{"1.0": "' + "a" * 64 + '"}\n', encoding="utf-8")
    elif case == "dirty":
        (here / "checklist.json").write_text((here / "checklist.json").read_text(encoding="utf-8") + " ", encoding="utf-8")
    else:
        (here / "CHANGELOG.md").write_text("# 기록\n", encoding="utf-8")
    before = {p.name: p.read_bytes() for p in here.iterdir() if p.is_file()}
    with pytest.raises(tool.FreezeError, match=re.escape(msg)):
        tool.freeze(version, here)
    assert {p.name: p.read_bytes() for p in here.iterdir() if p.is_file()} == before
