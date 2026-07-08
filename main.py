"""Render startCommand(`uvicorn main:app`) 하위 호환 shim — 실체는 app/main.py.

배포 설정(render.yaml·Render 대시보드)의 startCommand 를 `uvicorn app.main:app` 으로
전환하고 배포 1사이클 검증(리팩토링 계획 PR2)을 마치기 전까지 삭제 금지.
docs/REFACTORING_PLAN.md 참조.
"""
from app.main import app  # noqa: F401
