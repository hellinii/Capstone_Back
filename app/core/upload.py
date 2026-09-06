"""app/core/upload.py — 업로드 파일 경계 가드 (확장자 · 크기 · 빈 파일)

세 업로드 라우터(analyze-columns / validate-data / evaluate)가 공유하는 **단일 검사
지점**이다. 종전에는 이 세 검사가 라우터마다 따로 있거나 아예 없었다.

- 확장자 검사가 analyze-columns 에만 있어 `.xlsx` 가 analyze 는 400, 나머지 둘은
  파싱 예외를 타고 422 로 거절됐다(G-08).
- 크기 상한은 어디에도 없어 26 MB CSV 가 200 으로 통과했고, 그 5.33 초 동안
  단일 워커가 묶여 `/health` 응답이 5,032 ms 까지 밀렸다(G-04a, 실서버 실측).

**상한이 막는 것과 막지 못하는 것.** FastAPI 는 핸들러 진입 전에 멀티파트 본문을
이미 수신해 스풀 파일에 쓴다. 따라서 이 가드는 *수신* 을 막지 못하고 *처리* 를
막는다. 실측된 피해(pandas 파싱 CPU 점유 + DataFrame 메모리 증폭)는 처리 단계에서
발생하므로 이 지점이 올바른 차단점이다. 수신 자체를 끊으려면 프록시/미들웨어 층이
필요하고 그것은 이 가드의 범위가 아니다.

상호작용
- 의존(import): fastapi(UploadFile, HTTPException)
- 사용처: app.analysis.router / app.analysis.validation_router / app.evaluation.router
"""
from fastapi import HTTPException, UploadFile

# CSV/JSON 만 허용 — core.parsing.parse_file_content 이 실제로 다루는 두 형식.
ALLOWED_EXTENSIONS = {"csv", "json"}

# 업로드 상한 20 MiB. 근거: 로컬(M-series) 1워커에서 26 MB CSV 처리가 5.33 초 동안
# 이벤트 루프를 막았고 10.9 MB CSV 의 RSS 델타가 약 200 MB 였다. Render free 는
# 0.1 CPU / 512 MB 라 로컬보다 훨씬 느리다. 발급 본문 상한(issuance/schemas.py 의
# MAX_CONTENT_BYTES = 1 MiB)과는 별개 값이다 — 그쪽은 DB 에 영구 저장되는 JSON 이고
# 이쪽은 일회성 데이터셋이다.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

# 상한 초과분을 메모리에 쌓지 않기 위해 청크 단위로 읽으며 누적량을 검사한다.
_CHUNK_SIZE = 1024 * 1024


def _format_bytes(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB"


def extract_extension(filename: str) -> str:
    """파일명에서 소문자 확장자를 뽑는다. 확장자가 없으면 빈 문자열."""
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


async def read_upload_guarded(file: UploadFile) -> tuple[str, bytes]:
    """업로드 파일을 검사하며 읽어 (filename, content) 를 돌려준다.

    검사 순서와 응답은 세 라우터에서 동일하다:
      1. 확장자가 CSV/JSON 이 아니면 400
      2. 누적 바이트가 MAX_UPLOAD_BYTES 를 넘으면 413 (넘는 순간 읽기 중단)
      3. 내용이 비어 있으면 400
    """
    filename = file.filename or ""

    ext = extract_extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다: .{ext or '(없음)'}. CSV 또는 JSON 파일을 업로드해주세요.",
        )

    # 모듈 속성으로 조회한다 — 테스트가 상한을 monkeypatch 로 낮출 수 있게 하기 위함.
    limit = MAX_UPLOAD_BYTES
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"파일이 너무 큽니다. 최대 {_format_bytes(limit)} 까지 업로드할 수 있습니다."
                ),
            )
        chunks.append(chunk)

    if total == 0:
        raise HTTPException(status_code=400, detail="빈 파일은 처리할 수 없습니다.")

    return filename, b"".join(chunks)
