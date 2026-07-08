"""애플리케이션 패키지.

빈 파일로 유지할 것 — 어떤 재수출도 추가 금지.

여기에 `from app.main import app` 같은 편의 재수출을 넣으면 `import app` 시점에
load_dotenv() 보다 먼저 app.core.database 가 로드되어, 로컬 .env 의 DATABASE_URL 이
무시된 채 SQLite 기본값으로 조용히 폴백한다 (app/main.py 의 import 순서 주석 참조).
"""
