"""app/issuance/bootstrap.py — 발급 도메인 부트스트랩(기본 기관 시드)

기본 수행기관(Organization) 데이터와 시드 함수를 발급 도메인에 둔다. 기존에는 core.database 가
이 도메인 데이터를 품어 core→도메인 계층 역전 + 순환 회피용 지연 import 가 필요했으나,
issuance 로 옮겨 정상 import 로 정리했다(동작 불변).

상호작용
- 의존(import): app.core.database(SessionLocal), app.issuance.models(Organization)
- 사용처: app.main(lifespan 에서 seed_organization 호출)
"""
from app.core.database import SessionLocal
from app.issuance.models import Organization

# ── 기관(organization) 시드 기본값 — 현 프론트 DEFAULT_PERFORMER 와 일치 ───────────
DEFAULT_ORGANIZATION = {
    "id": 1,
    # 이 값이 수행기관의 **정본**이다(ISSUES.md F-10). 프론트
    # `src/lib/report/reportConstants.ts` 의 DEFAULT_PERFORMER 는 발급 전 초안 화면에만
    # 쓰이는 자리표시자이며 같은 값을 갖는다 — 별도 배포 단위라 사본을 피할 수 없다.
    # **기관 정보를 바꿀 때는 여기를 먼저 고치고 프론트 자리표시자를 맞춘다.**
    "org_name": "한국 AI 인증원",
    "department": "평가부",
    "evaluator": "자동 평가 엔진",
    "contact": "—",
    "address": None,
}


def seed_organization() -> None:
    """organization 이 비어 있으면 기본 기관 1행 INSERT(singleton, id=1)."""
    db = SessionLocal()
    try:
        if db.get(Organization, 1) is None:
            db.add(Organization(**DEFAULT_ORGANIZATION))
            db.commit()
    finally:
        db.close()
