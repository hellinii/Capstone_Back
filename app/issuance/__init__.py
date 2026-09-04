"""issuance 도메인 — 성적서 번호를 채번·발급·재발급하고 저장한다(유일하게 DB 사용).
요청 흐름의 4단계(발급).

파일 목차
- router.py       : 발급/조회 API (/api/organization, /api/reports/*) — 얇은 HTTP
- service.py      : 채번·발급·재발급 트랜잭션 로직(동시성 처리)
- models.py       : ORM 테이블(Organization 1:N Report 1:N Issuance)
- serializers.py  : ORM → 응답 스키마 직렬화(presenter)
- bootstrap.py    : 기본 수행기관 시드(seed_organization) — main 시작 시 호출
- schemas.py      : 기관·발급 요청/응답 스키마
"""
