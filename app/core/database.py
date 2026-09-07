"""database.py — SQLAlchemy 엔진 / 세션 / Base 및 get_db 의존성.

발급 메타(조직·성적서·발급차수)를 저장하는 SQLite DB (설계 문서 §2, §8).
- 파일: data/app.db (.gitignore). 단일 백엔드·낮은 동시성 → SQLite 로 충분.
- ORM 추상화로 추후 PostgreSQL 전환 용이.
- DATABASE_URL 환경변수로 경로 재정의 가능(테스트는 인메모리/임시 파일 DB 사용).
"""
import logging
import os

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# ── 엔진 ──────────────────────────────────────────────────────────────────────
# __file__ = <루트>/app/core/database.py → 상위 3단계가 레포 루트.
# 이 모듈이 app/core/ 로 이동했으므로 기본 SQLite 는 <루트>/data/app.db 를 가리키도록
# 루트 기준으로 재앵커한다(.gitignore 의 data/app.db 패턴과 일치, 대소문자도 소문자 통일).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_DB_PATH = os.path.join(_PROJECT_ROOT, "data", "app.db")
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{_DEFAULT_DB_PATH}")

# Heroku/Render 스타일 URL 은 legacy "postgres://" 스킴을 쓰는데 SQLAlchemy 2.0 은
# 이를 거부한다. (Neon 은 postgresql:// 을 주지만 방어적으로 정규화.)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = "postgresql://" + DATABASE_URL[len("postgres://"):]

_IS_SQLITE = DATABASE_URL.startswith("sqlite")


def describe_backend(database_url: str) -> tuple[str, bool]:
    """(백엔드 이름, 영속 여부). 연결 문자열 자체는 절대 밖으로 내보내지 않는다."""
    if database_url.startswith("sqlite"):
        return "sqlite", False
    return "postgresql", True


def warn_if_volatile(database_url: str) -> None:
    """휘발성 백엔드로 내려갔으면 기동 시점에 경고를 남긴다 (ISSUES.md G-07).

    하드 실패 가드는 기본 꺼짐이 옳다(아래 참조). 그렇다고 강등이 **아무 흔적도 남기지
    않아서는** 안 된다 — 1차 라운드가 성적서 원본을 DB 에 넣은 뒤로 휘발성 DB 는 채번
    중복만이 아니라 **발급된 성적서 자체를 잃는다.**
    """
    name, persistent = describe_backend(database_url)
    if not persistent:
        logger.warning(
            "DATABASE_URL 이 없거나 sqlite 입니다(backend=%s). 재시작하면 채번 시퀀스와 "
            "발급된 성적서 보관본이 사라집니다. 프로덕션이라면 영속 DB 를 연결하세요.",
            name,
        )


def assert_persistent_backend(database_url: str, require: bool) -> None:
    """영속 DB 를 요구하는 환경에서 휘발성 SQLite 로의 조용한 강등을 막는다(G-07).

    `DATABASE_URL` 이 없으면 이 모듈은 예외 없이 로컬 SQLite 로 내려가고 기동을
    계속한다. 그 상태에서는 채번 시퀀스가 재시작마다 초기화돼 **성적서 번호가
    중복 발급**된다.

    가드는 기본으로 꺼져 있다. 무조건 켜면 위험하기 때문이다 — 지금 프로덕션이
    SQLite 로 돌고 있다면 켜는 순간 '조용히 열화된 채 살아 있는 서비스'가
    '기동 실패로 죽은 서비스'가 된다. 코드만 봐서는 어느 경우인지 알 수 없으므로,
    `/health` 진단(DIAG=1)으로 실제 백엔드를 확인한 뒤 켜는 순서를 따른다.
    """
    if require and database_url.startswith("sqlite"):
        raise RuntimeError(
            "REQUIRE_PERSISTENT_DB=1 인데 DATABASE_URL 이 없거나 sqlite 입니다. "
            "휘발성 DB 로 기동하면 채번 시퀀스가 초기화되어 성적서 번호가 중복 발급됩니다."
        )

# 엔진을 만들기 전에 검사한다 — postgres URL 이면 create_engine 이 드라이버를 요구하므로
# 가드가 그 뒤에 있으면 진단 메시지 대신 ModuleNotFoundError 가 먼저 나온다.
assert_persistent_backend(DATABASE_URL, require=os.getenv("REQUIRE_PERSISTENT_DB") == "1")
warn_if_volatile(DATABASE_URL)

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
