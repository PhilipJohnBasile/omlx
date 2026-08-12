# SPDX-License-Identifier: Apache-2.0
"""Tests for the external GGUF engine (DS4/DwarfStar via ds4-server)."""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from omlx.engine.external_gguf import ExternalGGUFEngine, ExternalGGUFError
from omlx.model_discovery import discover_models


def test_discovery_registers_gguf_as_external():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        gguf = d / "DeepSeek-V4-Flash-0731-q2.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * 64)
        models = discover_models(d)
        assert "DeepSeek-V4-Flash-0731-q2" in models
        m = models["DeepSeek-V4-Flash-0731-q2"]
        assert m.engine_type == "external_gguf"
        assert m.model_type == "llm"
        assert m.model_path == str(gguf)


def test_engine_requires_ds4_server_binary(monkeypatch):
    os.environ.pop("OMLX_DS4_SERVER_BIN", None)
    with (
        patch("omlx.engine.external_gguf.shutil.which", return_value=None),
        pytest.raises(ExternalGGUFError),
    ):
        ExternalGGUFEngine(model_path="/tmp/x.gguf")


def test_engine_instantiation_and_stats(monkeypatch):
    monkeypatch.setenv("OMLX_DS4_SERVER_BIN", "/usr/bin/false")
    e = ExternalGGUFEngine(model_path="/tmp/x.gguf")
    assert e.model_name == "/tmp/x.gguf"
    assert e.model_type == "external_gguf"
    stats = e.get_stats()
    assert stats["engine"] == "external_gguf"
    assert stats["started"] is False


def test_engine_start_spawns_binary_and_waits_ready(monkeypatch):
    monkeypatch.setenv("OMLX_DS4_SERVER_BIN", "/usr/bin/false")
    e = ExternalGGUFEngine(model_path="/tmp/x.gguf", port=33123)
    # /bin/false exits immediately -> startup must raise, not hang
    with pytest.raises(ExternalGGUFError):
        asyncio.run(e.start())


def test_engine_render_messages():
    os.environ["OMLX_DS4_SERVER_BIN"] = "ds4-server"
    e = ExternalGGUFEngine(model_path="/tmp/x.gguf")
    rendered = e._render_messages([
        {"role": "system", "content": "be concise"},
        {"role": "user", "content": "hi"},
    ])
    assert "system: be concise" in rendered
    assert "user: hi" in rendered


@pytest.mark.asyncio
async def test_chat_proxies_and_maps_response(monkeypatch):
    monkeypatch.setenv("OMLX_DS4_SERVER_BIN", "/usr/bin/false")
    e = ExternalGGUFEngine(model_path="/tmp/x.gguf", port=33124)
    # stub start (no real subprocess) and _post (fake ds4-server response)
    e.start = AsyncMock()
    fake_body = json.dumps({
        "choices": [{"text": "hello world", "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }).encode()
    e._post = AsyncMock(return_value=fake_body)
    out = await e.chat(
        [{"role": "user", "content": "say hello"}],
        max_tokens=8, temperature=0,
    )
    assert out.text == "hello world"
    assert out.completion_tokens == 2
    assert out.prompt_tokens == 5
    # _post must have hit the completions endpoint with the rendered prompt
    args = e._post.call_args
    assert args[0][0] == "/v1/completions"
    payload = args[0][1]
    assert isinstance(payload, dict)
    assert "say hello" in payload["prompt"]


@pytest.mark.asyncio
async def test_generate_rejects_token_ids(monkeypatch):
    monkeypatch.setenv("OMLX_DS4_SERVER_BIN", "/usr/bin/false")
    e = ExternalGGUFEngine(model_path="/tmp/x.gguf")
    e.start = AsyncMock()
    with pytest.raises(ExternalGGUFError):
        await e.generate([1, 2, 3])


@pytest.mark.asyncio
async def test_stream_generate_parses_sse(monkeypatch):
    monkeypatch.setenv("OMLX_DS4_SERVER_BIN", "/usr/bin/false")
    e = ExternalGGUFEngine(model_path="/tmp/x.gguf", port=33125)
    e.start = AsyncMock()
    sse = (
        b'data: {"choices":[{"delta":{"text":"hel"}}]}\n'
        b'data: {"choices":[{"delta":{"text":"lo"}}]}\n'
        b'data: [DONE]\n'
    )
    e._post = AsyncMock(return_value=sse)
    chunks = [c.text async for c in e.stream_generate("hi", max_tokens=4)]
    assert chunks == ["hel", "lo", ""]  # final finished=True chunk
