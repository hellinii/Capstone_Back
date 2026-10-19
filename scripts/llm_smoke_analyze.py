"""
llm_smoke_analyze.py — 컬럼 매핑 + 메타데이터 추출 수동 스모크 스크립트.

pytest 테스트가 아니라 실제 OPENAI_API_KEY 로 LLM 분석을 왕복 확인하는 CLI.
레포 루트에서: python scripts/llm_smoke_analyze.py [파일 경로 ...]
파일을 주면 그 파일만, 안 주면 tests/data 아래 전부. 모델은 MAPPING_MODEL(없으면 코드 기본값).
파일마다 걸린 시간을 찍는다(운영 클라이언트 시간 제한 45초 안인지 본다).
"""

import asyncio
import os
import sys
import time

# 레포 루트를 sys.path 에 추가 (app.* 절대 import 를 위해)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import AsyncOpenAI

from app.core.schemas import TaskType
from app.core.parsing import parse_file_content
from app.analysis.llm_mapper import analyze_columns_with_llm, mapping_model

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "data")


def detect_task_type(filepath: str) -> TaskType:
    lower = filepath.lower()
    if "binary" in lower:
        return TaskType.binary
    if "multiclass" in lower:
        return TaskType.multiclass
    if "multilabel" in lower:
        return TaskType.multilabel
    raise ValueError(f"task_type 판별 불가: {filepath}")


def collect_test_files() -> list[tuple[str, TaskType]]:
    result = []
    for root, _, files in os.walk(DATA_DIR):
        for fname in sorted(files):
            ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
            if ext not in ("csv", "json"):
                continue
            full_path = os.path.join(root, fname)
            try:
                result.append((full_path, detect_task_type(full_path)))
            except ValueError as e:
                print(f"  ⚠️  {e}")
    return result


async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ .env 파일에 OPENAI_API_KEY를 설정해주세요!")
        return

    client = AsyncOpenAI(api_key=api_key, max_retries=0)
    if len(sys.argv) > 1:
        test_files = [(os.path.abspath(p), detect_task_type(p)) for p in sys.argv[1:]]
    else:
        test_files = collect_test_files()
    print(f"🤖 매핑 모델: {mapping_model()}")

    print(f"📁 총 {len(test_files)}개 파일 테스트 시작\n")
    success, fail = 0, 0

    for full_path, task_type in test_files:
        rel_path = os.path.relpath(full_path, os.path.dirname(__file__))
        filename = os.path.basename(full_path)

        print(f"── {rel_path} ({task_type.value})")

        try:
            with open(full_path, "rb") as f:
                content = f.read()

            columns, df = parse_file_content(content, filename)
            started = time.perf_counter()
            result = await analyze_columns_with_llm(client, task_type, columns, df)
            print(f"     ⏱  {time.perf_counter() - started:.2f}초")

            # 컬럼 매핑 출력
            for m in result.column_mappings:
                print(f"     {m.column:<25} → {m.role.value}")

            # 메타데이터 출력
            meta = result.metadata
            if task_type == TaskType.binary:
                ambiguous = " ⚠️ 자동 판단 불확실 (사용자 확인 필요)" if meta.positive_class_ambiguous else ""
                print(f"     📌 Positive class: {meta.positive_class!r}  /  Negative: {meta.negative_class!r}{ambiguous}")
            elif task_type == TaskType.multiclass:
                print(f"     📌 감지된 클래스 ({len(meta.detected_classes)}개): {meta.detected_classes}")
            elif task_type == TaskType.multilabel:
                print(f"     📌 감지된 레이블 ({len(meta.detected_labels)}개): {meta.detected_labels}")

            print(f"     📊 분포: { {k: v for k, v in list(meta.class_distribution.items())[:5]} }" +
                  (" ..." if len(meta.class_distribution) > 5 else ""))

            success += 1

        except Exception as e:
            print(f"     ❌ 오류: {e}")
            fail += 1

        print()

    print(f"{'='*50}")
    print(f"✅ 성공: {success}개   ❌ 실패: {fail}개   합계: {len(test_files)}개")


if __name__ == "__main__":
    asyncio.run(main())
