import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from FastApi_stream_ui.api.pipeline_model import Question, Answer
from FastApi_stream_ui.api import pipeline_fapi_askllm


@pytest.mark.asyncio
async def test_ask_llm_returns_answer(monkeypatch):
    # Arrange: create a fake response object
    fake_response = AsyncMock()
    fake_response.choices = [AsyncMock()]
    fake_response.choices[0].message.content = "Mocked answer"

    # Patch the OpenAI client call
    async def fake_create(*args, **kwargs):
        return fake_response

    monkeypatch.setattr(
        pipeline_fapi_askllm.client.chat.completions, "create", fake_create
    )

    q = Question(question="What is RAG in one sentence?")

    # Act
    ans = await pipeline_fapi_askllm.ask_llm(q)

    # Assert
    assert isinstance(ans, Answer)
    assert ans.question == "What is RAG in one sentence?"
    assert ans.text == "Mocked answer"
    assert ans.cost_usd == 0.0001
