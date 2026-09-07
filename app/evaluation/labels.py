"""app/evaluation/labels.py — 라벨 표현형의 단일 출처.

"셀 하나를 어떤 라벨 목록으로 읽는가"와 "라벨을 어떤 순서로 늘어놓는가"를 정하는 규칙은
**한 곳에만** 있어야 한다. 종전에는 라벨을 만들어내는 파서가 넷이었다(ISSUES.md D-04):

  (1) `evaluation/preprocessor._parse_multilabel_value`  — literal_eval + '|' + ','
  (2) `evaluation/metrics/multilabel._parse_multilabel_col` — 같은 규칙(사본)
  (3) `analysis/metadata` 의 컬럼 고유값 수집          — `str()` 후 '|' **만**
  (4) `analysis/metadata` 의 멀티레이블 분포 집계       — `str()` 후 '|' **만**

그래서 `"a,b"` 는 (1)(2)에서 라벨 **2개**, (3)(4)에서 라벨 **1개**였다. (3)(4)의 산출물은
프론트의 `detected_labels`·`class_distribution` 이 되어 성적서에 인쇄되므로, 한 문서 안에
서로 다른 라벨 집합이 실렸다.

## 왜 문자열로 통일하는가 — 그리고 왜 그것만으로는 모자란가

(1)의 `ast.literal_eval` 분기는 `'[1, 2]'` 를 **int 리스트**로, `'3'` 을 **str** 로 만든다.
같은 컬럼에서 두 타입이 섞이면 `MultiLabelBinarizer` 가 `'<' not supported` 로 죽는다.
그보다 조용한 피해가 더 크다 — `service.py` 의 `{str(k): v}` 가 int 키 `1` 과 str 키 `'1'`
을 같은 키로 뭉개 **나중 값이 앞 값을 덮어쓴다.** 실측: 합 6 인 분포가 합 4 로 줄어든
채 성적서에 인쇄됐다.

그래서 파서가 라벨을 **항상 문자열로** 내놓는다. 다만 문자열로 만드는 순간 정렬이
사전순이 되어 `['1','10','2','3']` 처럼 클래스 순서가 조용히 뒤집힌다. 그 역전은 이미
절반 일어나 있었다 — `metadata` 는 `sorted(str)` 로 `['1','10','2']`, `M21` 의 라벨은
`sorted(np.unique(...))` 로 `[1,2,10]` 이라 **한 성적서 안에 클래스 순서가 두 벌**이었다.
`sort_labels` 가 "전부 숫자면 수치 순서"로 두 순서를 하나로 합친다.

상호작용
- 의존(import): 표준 라이브러리만
- 사용처: evaluation.preprocessor, evaluation.metrics.multilabel, analysis.metadata,
  evaluation.service(분포 정규화)
"""
from typing import Any, Iterable, List
import ast


def normalize_label(value: Any) -> str:
    """라벨 하나를 표준 표현형(문자열)으로.

    숫자는 소수점 없는 정수로 만든다 — pandas 가 정수 컬럼을 float 로 읽으면 라벨이
    `'1.0'` 이 되어 `'1'` 과 다른 라벨이 된다.
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def parse_label_cell(item: Any) -> List[str]:
    """멀티레이블 셀 하나 → 라벨 목록(항상 문자열).

    받는 표기: 리스트, `"A|B"`, `"A,B"`, `"['A','B']"`, 단일 라벨 `"A"`.
    빈 셀은 '해당 라벨 없음'이라는 정상 입력이므로 빈 목록이다.
    """
    if item is None:
        return []
    if isinstance(item, (list, tuple, set)):
        return [normalize_label(x) for x in item if normalize_label(x)]
    if isinstance(item, str):
        text = item.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return [normalize_label(x) for x in parsed if normalize_label(x)]
        except (ValueError, SyntaxError):
            pass
        separator = "|" if "|" in text else ","
        return [normalize_label(x) for x in text.split(separator) if normalize_label(x)]
    label = normalize_label(item)
    return [label] if label else []


def sort_labels(labels: Iterable[Any]) -> List[str]:
    """라벨을 표준 순서로 늘어놓는다.

    **전부 숫자면 수치 순서를 유지한다.** 단순 문자열 정렬은 `['1','10','2','3']` 을
    만들어 성적서의 클래스 순서를 조용히 뒤집는다 — 독자가 클래스 1 다음에 10 이 오는
    표를 보게 된다. 하나라도 숫자가 아니면 사전순으로 되돌린다(혼합 정렬은 비교 자체가
    성립하지 않는다).
    """
    normalized = [normalize_label(l) for l in labels]
    unique = list(dict.fromkeys(normalized))
    try:
        return sorted(unique, key=lambda x: (float(x), x))
    except (TypeError, ValueError):
        return sorted(unique)


def normalize_distribution(distribution: dict) -> dict:
    """클래스 분포의 키를 표준 표현형으로 바꾸되 **개수를 합친다**.

    `{str(k): v}` 로 단순 변환하면 int 키 `1` 과 str 키 `'1'` 이 같은 키가 되면서
    나중 값이 앞 값을 **덮어쓴다** — 예외 없이 카운트가 사라진다(ISSUES.md D-04).
    같은 키로 모이면 더해야 한다.
    """
    merged: dict[str, int] = {}
    for key, count in distribution.items():
        merged[normalize_label(key)] = merged.get(normalize_label(key), 0) + int(count)
    return {label: merged[label] for label in sort_labels(merged.keys())}
