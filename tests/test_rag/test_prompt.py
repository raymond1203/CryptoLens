from unittest.mock import MagicMock

from src.rag.prompt import SYSTEM_PROMPT, build_rag_prompt


class TestBuildRagPrompt:
    def test_returns_user_message_with_context(self):
        point = MagicMock()
        point.payload = {
            "title": "Bitcoin ATH",
            "source": "coindesk",
            "url": "https://coindesk.com/article",
            "text": "Bitcoin reached a new all-time high.",
        }

        messages = build_rag_prompt("비트코인 최신 뉴스", [point])

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert "비트코인 최신 뉴스" in content
        assert "Bitcoin ATH" in content
        assert "coindesk" in content
        assert "https://coindesk.com/article" in content

    def test_handles_missing_payload_fields(self):
        point = MagicMock()
        point.payload = {}

        messages = build_rag_prompt("질문", [point])

        content = messages[0]["content"]
        assert "제목 없음" in content
        assert "출처 없음" in content

    def test_multiple_context_points(self):
        points = []
        for i in range(3):
            p = MagicMock()
            p.payload = {"title": f"Article {i}", "source": "test", "text": f"Content {i}"}
            points.append(p)

        messages = build_rag_prompt("질문", points)

        content = messages[0]["content"]
        assert "[1]" in content
        assert "[2]" in content
        assert "[3]" in content


class TestSystemPrompt:
    def test_contains_disclaimer(self):
        assert "투자 조언이 아니" in SYSTEM_PROMPT
        assert "법률 자문을 대체하지 않습니다" in SYSTEM_PROMPT

    def test_contains_korean_response_instruction(self):
        assert "한국어" in SYSTEM_PROMPT

    def test_contains_citation_rule(self):
        assert "출처" in SYSTEM_PROMPT
