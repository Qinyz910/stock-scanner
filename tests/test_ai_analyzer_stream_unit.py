import asyncio
import json
import types
from typing import List

import pandas as pd
import httpx

from services.ai_analyzer import AIAnalyzer


class FakeStreamResponse:
    def __init__(self, lines: List[str], status_code: int = 200):
        self._lines = lines
        self.status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_text(self):
        # Yield each line as a chunk, with newline like real SSE
        for line in self._lines:
            yield line + "\n"

    async def aread(self):
        return json.dumps({"error": {"message": "bad request"}}).encode()


class FakeResponse:
    def __init__(self, status_code: int = 200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {"choices": [{"message": {"content": "fallback result"}}]}

    def json(self):
        return self._json_data


class FakeAsyncClient:
    def __init__(self, timeout=None, stream_lines: List[str] = None, stream_status: int = 200, post_json=None):
        self._timeout = timeout
        self._stream_lines = stream_lines or []
        self._stream_status = stream_status
        self._post_json = post_json

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream(self, method, url, json=None, headers=None):
        return FakeStreamResponse(self._stream_lines, status_code=self._stream_status)

    async def post(self, url, json=None, headers=None):
        return FakeResponse(200, json_data=(self._post_json or {"choices": [{"message": {"content": "fallback result"}}]}))


def make_df(rows: int = 20) -> pd.DataFrame:
    data = {
        "MA5": [10 + i * 0.1 for i in range(rows)],
        "MA20": [9 + i * 0.1 for i in range(rows)],
        "Volatility": [1.0 for _ in range(rows)],
        "Volume_Ratio": [1.2 for _ in range(rows)],
        "RSI": [50 for _ in range(rows)],
        "MACD": [0.5 for _ in range(rows)],
        "MACD_Signal": [0.4 for _ in range(rows)],
        "Close": [100 + i for i in range(rows)],
        "Change": [0.01 for _ in range(rows)],
    }
    return pd.DataFrame(data)


async def collect_chunks(analyzer: AIAnalyzer, df: pd.DataFrame, stream: bool = True):
    chunks = []
    completed = False
    async for item in analyzer.get_ai_analysis(df, "TEST", market_type="A", stream=stream):
        obj = json.loads(item)
        if "ai_analysis_chunk" in obj:
            chunks.append(obj["ai_analysis_chunk"])
        if obj.get("status") == "completed":
            completed = True
    return chunks, completed


def test_stream_normal_chunks(monkeypatch):
    lines = [
        'data: {"choices":[{"delta":{"role":"assistant"}}]}',
        'data: {"choices":[{"delta":{"content":"第一段"}}]}',
        'data: {"choices":[{"delta":{"content":" 第二段"}}]}',
        'data: {"choices":[{"delta":{"content":" 结束。"}}]}',
        'data: [DONE]'
    ]

    def fake_client_factory(*args, **kwargs):
        return FakeAsyncClient(stream_lines=lines)

    monkeypatch.setattr(httpx, "AsyncClient", fake_client_factory)

    analyzer = AIAnalyzer(custom_api_url="http://fake", custom_api_key="k", custom_api_model="m")
    df = make_df()
    chunks, completed = asyncio.run(collect_chunks(analyzer, df, stream=True))

    assert completed is True
    assert "第一段" in "".join(chunks)
    assert "第二段" in "".join(chunks)


def test_stream_zero_chunks_then_fallback(monkeypatch):
    lines = [
        'data: {"choices":[{"delta":{"role":"assistant"}}]}',
        'data: [DONE]'
    ]
    post_json = {"choices": [{"message": {"content": "补救分析内容"}}]}

    def fake_client_factory(*args, **kwargs):
        return FakeAsyncClient(stream_lines=lines, post_json=post_json)

    monkeypatch.setattr(httpx, "AsyncClient", fake_client_factory)

    analyzer = AIAnalyzer(custom_api_url="http://fake", custom_api_key="k", custom_api_model="m")
    df = make_df()
    chunks, completed = asyncio.run(collect_chunks(analyzer, df, stream=True))

    assert completed is True
    assert any("补救分析内容" in c for c in chunks)


def test_stream_tool_call_frames_then_fallback(monkeypatch):
    lines = [
        'data: {"choices":[{"delta":{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"foo","arguments":"{}"}}]}}]}',
        'data: [DONE]'
    ]
    post_json = {"choices": [{"message": {"content": "工具调用后补救内容"}}]}

    def fake_client_factory(*args, **kwargs):
        return FakeAsyncClient(stream_lines=lines, post_json=post_json)

    monkeypatch.setattr(httpx, "AsyncClient", fake_client_factory)

    analyzer = AIAnalyzer(custom_api_url="http://fake", custom_api_key="k", custom_api_model="m")
    df = make_df()
    chunks, completed = asyncio.run(collect_chunks(analyzer, df, stream=True))

    assert completed is True
    assert any("工具调用后补救内容" in c for c in chunks)
