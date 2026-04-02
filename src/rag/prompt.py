from qdrant_client import models

SYSTEM_PROMPT = """\
당신은 CryptoLens, 크립토 시장 분석 AI 어시스턴트입니다.

## 역할
- 영어/한국어 크립토 소스를 분석하여 한국어로 인사이트를 제공합니다.
- 검색된 컨텍스트 내의 정보만 사용하여 답변합니다.

## 출처 인용 규칙
- 모든 주장에 출처를 명시합니다: [출처명](URL)
- 컨텍스트에 없는 정보는 "확인이 필요합니다"라고 명시합니다.

## 응답 규칙
- 항상 한국어로 답변합니다.
- 간결하고 구조화된 답변을 제공합니다.

## 면책 고지
- 응답 마지막에 반드시 다음 문구를 포함합니다:

---
⚠️ 본 정보는 투자 조언이 아니며, 전문 법률 자문을 대체하지 않습니다. \
투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다.\
"""


def build_rag_prompt(query: str, context_points: list[models.ScoredPoint]) -> list[dict]:
    """검색 결과를 컨텍스트로 주입한 메시지 리스트를 생성한다."""
    context_block = _format_context(context_points)

    user_content = f"""다음은 검색된 관련 문서입니다:

{context_block}

---
질문: {query}"""

    return [{"role": "user", "content": user_content}]


def _format_context(points: list[models.ScoredPoint]) -> str:
    """검색 결과를 번호 매긴 텍스트 블록으로 포맷한다."""
    blocks: list[str] = []
    for i, point in enumerate(points, 1):
        payload = point.payload or {}
        title = payload.get("title", "제목 없음")
        source = payload.get("source", "출처 없음")
        url = payload.get("url", "")
        text = payload.get("text", "")

        header = f"[{i}] {title} ({source})"
        if url:
            header += f"\n    URL: {url}"
        blocks.append(f"{header}\n{text}")

    return "\n\n".join(blocks)
