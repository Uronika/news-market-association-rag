import os

import anyio

from app.config import Settings
from app.llm import DeepSeekLLMClient
from app.secrets import save_deepseek_api_key


def test_save_deepseek_key_updates_current_process_env(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("LLM_PROVIDER=mock\n", encoding="utf-8")
    monkeypatch.setattr("app.secrets.ENV_PATH", env_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER", "mock")

    save_deepseek_api_key("  sk-test-value  ")

    assert os.environ["DEEPSEEK_API_KEY"] == "sk-test-value"
    assert os.environ["LLM_PROVIDER"] == "deepseek"
    assert "DEEPSEEK_API_KEY=sk-test-value" in env_path.read_text(encoding="utf-8")


def test_deepseek_client_uses_timeout_and_proxy(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json, headers):
            captured["url"] = url
            captured["authorization"] = headers["Authorization"]
            return FakeResponse()

    monkeypatch.setattr("app.llm.httpx.AsyncClient", FakeClient)
    settings = Settings()
    settings.deepseek_api_key = ""
    settings.deepseek_request_timeout_seconds = 91
    settings.proxy_url = "http://127.0.0.1:7890"
    client = DeepSeekLLMClient(settings, api_key_override=' "sk-quoted" ')

    result = anyio.run(
        client.generate,
        "question",
        [{"title": "news", "url": "https://example.com"}],
        {},
        {},
        {},
    )

    assert result["provider"] == "deepseek"
    assert captured["timeout"] == 91
    assert captured["proxy"] == "http://127.0.0.1:7890"
    assert captured["authorization"] == "Bearer sk-quoted"
