"""core — 모든 도메인이 공유하는 계약(schemas)과 인프라(database·parsing).

⚠️ 빈 로직 패키지로 유지 — 재수출 금지. database 의 조기 로드는 main.py 의
load_dotenv() 순서 불변식을 깨뜨린다(app/__init__.py 주석 참조).

파일 목차
- schemas.py   : 전 도메인 공유 계약(enum·역할 규칙·TC 요건·인계 모델 ColumnMapping/DataMetadata)
- database.py  : SQLAlchemy 엔진/세션/Base/get_db/init_db (SQLite 로컬·Postgres 배포)
- parsing.py   : 업로드 파일(CSV/JSON) → DataFrame 파싱(공용 유틸)
"""
