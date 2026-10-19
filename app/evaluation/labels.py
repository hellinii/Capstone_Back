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
import math
import re


def label_text(value: Any) -> str:
    """라벨 하나의 숫자 표기만 맞춘 문자열 — 앞뒤 공백은 **그대로** 둔다.

    정수 값 float(1.0)은 `'1'` 로 쓴다. 빈 칸 하나 때문에 pandas 가 정수 컬럼을 float 로
    읽어도 정수 컬럼('1')과 같은 표기가 된다. 공백까지 지우는 표준 표기는 normalize_label 이다.
    공백이 남은 원래 표기가 필요한 곳(preprocessor.label_notation_changes — 무엇이 합쳐졌는지 알리는 곳)이
    이 함수를 쓴다. 평가·검증의 클래스 비교는 normalize_label 이다(S2).
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    # 글자로 온 정수 소수('1.0'·'2.00')도 같은 표기다 — JSON 의 "1.0" 과 CSV 의 1 이 다른 클래스가 되지 않게.
    if isinstance(value, str) and _INTEGER_DECIMAL.match(value.strip()):
        # float 을 거치지 않는다 — 큰 정수('12345678901234567891.0')가 정밀도를 잃어 다른 값이 되지 않게.
        digits = re.sub(r"\.0+$", "", value.strip()).lstrip("+")
        if digits.lstrip("-").strip("0") == "":
            digits = "0"
        return value.replace(value.strip(), digits)
    return str(value)


_INTEGER_DECIMAL = re.compile(r"^[+-]?\d+\.0+$")


def normalize_label(value: Any) -> str:
    """라벨 하나를 표준 표현형(문자열)으로 — label_text 표기에서 앞뒤 공백까지 지운다.

    숫자는 소수점 없는 정수로 만든다 — pandas 가 정수 컬럼을 float 로 읽으면 라벨이
    `'1.0'` 이 되어 `'1'` 과 다른 라벨이 된다.
    """
    return label_text(value).strip()


def find_label(values: Iterable[Any], name: Any) -> Any | None:
    """표준 표기 `name` 에 해당하는 **원래 값**(dtype·앞뒤 공백 그대로)을 찾는다 — 없으면 None.

    화면·metadata 는 클래스를 normalize_label 표기('1'·'cat')로 주고받지만 데이터의 값은 1.0·' cat'
    일 수 있다. `y_true == 양성` 처럼 원래 값과 비교해야 하는 곳은 str() 로 맞춰 보지 말고 이 함수로
    되찾는다(빈 칸 섞인 정수 정답이 float 이 되어 '1' ≠ '1.0' 이 되던 문제).
    """
    if name is None:
        return None
    target = normalize_label(name)
    for v in values:
        if normalize_label(v) == target:
            return v
    return None


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


# ── 양성 클래스 기본값(M5) ────────────────────────────────────────────────────
# 사용자가 양성 클래스를 고르지 않았을 때 무엇을 양성으로 볼지 정하는 규칙은 이 함수 하나다. 종전에는
# 세 곳이 따로 정했다 — 자동 판단(analysis.metadata)은 모르는 두 문자열이면 **앞쪽**을, 지표 계산
# (metrics.binary)과 예측 파생(preprocessor)은 **뒤쪽**을 골라, 화면에 보인 양성과 성적서 수치의 양성이
# 달랐다. SPEC §0("문자면 알파벳 후순위")에 맞춰 뒤쪽으로 모았다.
#
# 순서: ① 숫자면 큰 값(1 > 0) ② 알려진 짝(yes/no·spam/ham·fraud/normal …)이면 정해진 값
#       ③ 그 밖은 표준 순서(sort_labels)의 뒤쪽 값 — 임시로 고른 것이라 ambiguous 로 알린다.
KNOWN_POSITIVE = frozenset({"1", "yes", "true", "positive", "pos", "spam", "malignant", "fraud", "1.0"})
KNOWN_NEGATIVE = frozenset({"0", "no", "false", "negative", "neg", "ham", "benign", "normal", "0.0"})


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def default_positive_class(values: Iterable[Any]) -> tuple[Any | None, Any | None, bool]:
    """정답 값들에서 기본 양성·음성 클래스를 고른다 → (양성, 음성, 사용자 확인 필요 여부).

    돌려주는 값은 **입력에 있던 원래 값**(dtype 그대로)이라 `y_true == 양성` 비교에 바로 쓴다.
    값 종류가 둘이 아니면 이진 판단이 안 되므로 음성은 None, ambiguous 는 True 이고, 양성은 표준
    순서의 뒤쪽 값을 임시로 준다(값이 하나도 없으면 None).
    """
    raw: dict[str, Any] = {}
    for v in values:
        if not _is_missing(v):
            raw.setdefault(normalize_label(v), v)
    if not raw:
        return None, None, True
    order = sort_labels(raw)
    if len(order) != 2:
        return raw[order[-1]], None, True

    first, last = order
    try:
        float(first), float(last)
        return raw[last], raw[first], False  # 숫자 — sort_labels 가 수치 순서라 뒤쪽이 큰 값
    except ValueError:
        pass
    lf, ll = first.lower(), last.lower()
    if lf in KNOWN_POSITIVE and ll in KNOWN_NEGATIVE:
        return raw[first], raw[last], False
    if ll in KNOWN_POSITIVE and lf in KNOWN_NEGATIVE:
        return raw[last], raw[first], False
    return raw[last], raw[first], True
