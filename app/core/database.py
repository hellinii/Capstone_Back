"""database.py — SQLAlchemy 엔진 / 세션 / Base 및 get_db 의존성.

발급 메타(조직·성적서·발급차수)를 저장하는 SQLite DB (설계 문서 §2, §8).
- 파일: data/app.db (.gitignore). 단일 백엔드·낮은 동시성 → SQLite 로 충분.
- ORM 추상화로 추후 PostgreSQL 전환 용이.
- DATABASE_URL 환경변수로 경로 재정의 가능(테스트는 인메모리/임시 파일 DB 사용).
"""
import os

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# ── 엔진 ──────────────────────────────────────────────────────────────────────
# __file__ = <루트>/app/core/database.py → 상위 3단계가 레포 루트.
# 이 모듈이 app/core/ 로 이동했으므로 기본 SQLite 는 <루트>/data/app.db 를 가리키도록
# 루트 기준으로 재앵커한다(.gitignore 의 data/app.db 패턴과 일치, 대소문자도 소문자 통일).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_DB_PATH = os.path.join(_PROJECT_ROOT, "data", "app.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{_DEFAULT_DB_PATH}")

# Heroku/Render 스타일 URL 은 legacy "postgres://" 스킴을 쓰는데 SQLAlchemy 2.0 은
# 이를 거부한다. (Neon 은 postgresql:// 을 주지만 방어적으로 정규화.)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = "postgresql://" + DATABASE_URL[len("postgres://"):]

_IS_SQLITE = DATABASE_URL.startswith("sqlite")

if _IS_SQLITE:
    _ENGINE_KWARGS: dict = {
        # check_same_thread=False: FastAPI 스레드풀에서 세션 사용 허용(요청별 세션이라 안전).
        "connect_args": {"check_same_thread": False},
    }
else:
    _ENGINE_KWARGS = {
        # Neon free tier 는 ~5분 유휴 시 compute suspend → 풀에 남은 연결이 죽는다.
        "pool_pre_ping": True,   # checkout 시 연결 검증, 죽었으면 투명하게 재연결
        "pool_recycle": 300,     # suspend 윈도우(5분)보다 오래된 연결은 선제 폐기
    }

engine = create_engine(DATABASE_URL, **_ENGINE_KWARGS)


def configure_sqlite(bind_engine) -> None:
    """SQLite 연결/트랜잭션 설정 — FK 강제 + BEGIN IMMEDIATE 쓰기 직렬화(설계 §4).

    pysqlite 의 암시적 BEGIN 을 끄고(isolation_level=None) 트랜잭션 시작 시 직접
    BEGIN IMMEDIATE 를 emit → 동시 채번/재발급이 교착(deadlock) 대신 RESERVED 락을
    즉시 잡고 대기(busy_timeout)한다. 대기 초과 시 OperationalError → 서비스가 재시도.
    (테스트도 이 함수를 재사용해 프로덕션과 동일한 잠금 동작을 검증한다.)
    """

    @event.listens_for(bind_engine, "connect")
    def _on_connect(dbapi_conn, _record):
        dbapi_conn.isolation_level = None  # pysqlite autobegin 비활성화 → 수동 BEGIN 제어
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")   # report/issuance FK 무결성
        cur.execute("PRAGMA busy_timeout=5000")  # 쓰기 락 대기(ms) — 즉시 BUSY 실패 방지
        cur.close()

    @event.listens_for(bind_engine, "begin")
    def _on_begin(conn):
        # 쓰기 직렬화: RESERVED 락을 즉시 획득 → 동시 채번 교착 대신 순번 대기.
        conn.exec_driver_sql("BEGIN IMMEDIATE")


if _IS_SQLITE:
    configure_sqlite(engine)


SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    """모든 ORM 모델의 선언적 베이스."""
    pass


def get_db():
    """FastAPI 의존성 — 요청 단위 DB 세션(응답 후 반드시 close)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── 기관(organization) 시드 기본값 — 현 프론트 DEFAULT_PERFORMER 와 일치 ───────────
DEFAULT_ORGANIZATION = {
    "id": 1,
    "org_name": "한국 AI 인증원",
    "department": "평가부",
    "evaluator": "자동 평가 엔진",
    "contact": "—",
    "address": None,
}


def init_db() -> None:
    """테이블 생성(이미 있으면 무시) — 마이그레이션 도구 없이 시작(설계 §8·§11)."""
    if _IS_SQLITE:
        path = DATABASE_URL.replace("sqlite:///", "")
        if path and path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # Base 에 테이블을 등록하기 위해 지연 import (models 가 이 모듈의 Base 를 import → 순환 방지).
    # ⚠️ 함수 내부 지연 import 유지 필수 — 최상단으로 올리면 database↔models 순환으로 깨진다.
    import app.issuance.models  # noqa: F401
    Base.metadata.create_all(bind=engine)


def seed_organization() -> None:
    """organization 이 비어 있으면 기본 기관 1행 INSERT(singleton, id=1)."""
    from app.issuance.models import Organization  # 지연 import (순환 방지) — 위치 유지

    db = SessionLocal()
    try:
        if db.get(Organization, 1) is None:
            db.add(Organization(**DEFAULT_ORGANIZATION))
            db.commit()
    finally:
        db.close()
