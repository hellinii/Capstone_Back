"""
narrative_prompt.py — LLM 서술 생성 프롬프트 + structured output 스키마.

analyzer.py 의 프롬프트/스키마 패턴을 따른다. 핵심 제약:
  - fact_sheet JSON 안의 숫자만 사용. 새 숫자 계산·반올림·백분율 환산·추정 금지.
  - latency 가 available=false 면 지연 수치를 언급하지 않는다.
  - conclusion.verdict 는 주어진 값을 그대로 echo (서버가 최종 강제).
  - benchmark 서술에는 source_note(기준 출처/성격)를 포함한다.
"""
import json



# 분류 유형의 성적서 이름(재료 목록 2절 "작업 유형 — 전 절, AI 예"). 한국어 이름은 이 셋으로 통일한다(W9,
# 2026-10-18 사용자 결정) — 프론트 성적서의 TASK_TYPE_LABELS 와 같은 글자다. 이미 발급된 성적서는 저장된
# 이름 그대로 두고, 새 서술 요청부터 이 이름이 AI 에 간다.
TASK_TYPE_LABELS = {"binary": "이진 분류", "multiclass": "다중 클래스 분류", "multilabel": "다중 레이블 분류"}


