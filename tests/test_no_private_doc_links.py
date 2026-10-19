"""저장소 문서와 코드가 저장소 밖의 개인 문서를 가리키지 않게 막는다(D2 — 프론트 noPrivateDocLinks.test.ts 와 같은 규칙).

설계 원문을 claude.ai 링크나 개인 폴더 경로(capstone_md_file)로 가리키면, 그 문서를 열 수 없는 팀원은
근거를 읽을 수 없다. 필요한 내용은 저장소 문서(docs/)에 옮겨 적는다.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRIVATE_LINK = re.compile(r"claude\.ai|claude docs|capstone_md_file", re.IGNORECASE)


def _files() -> list[Path]:
    files = [*ROOT.joinpath("docs").rglob("*.md"), ROOT / "README.md", *ROOT.joinpath("app").rglob("*.py")]
    return [p for p in files if p.is_file()]


def test_no_private_doc_links_in_docs_or_code():
    files = _files()
    # 빈 집합을 돌면 늘 통과한다 — 문서 5개 + app 코드가 실제로 잡히는지 먼저 본다.
    assert len([p for p in files if p.suffix == ".md"]) >= 5
    assert len(files) > 50
    hits = [
        f"{p.relative_to(ROOT)}:{i}"
        for p in files
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1)
        if PRIVATE_LINK.search(line)
    ]
    assert hits == []
