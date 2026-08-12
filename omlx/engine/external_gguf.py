"""External GGUF engine: serve DeepSeek V4 (and other) GGUFs via ds4-server.

oMLX does not load GGUF into MLX. For a registered ``external_gguf`` model
this engine spawns the DwarfStar ``ds4-server`` binary as a subprocess and
proxies OpenAI-compatible chat/generate requests to it. The server owns the
Metal kernels and quantization; oMLX owns the API surface and lifecycle.

Env contract (mirrors the DS4 project's server CLI):
- OMLX_DS4_SERVER_BIN: path to the ``ds4-server`` executable
  (default: ``ds4-server`` on PATH)
- OMLX_DS4_CTX: context tokens passed to ds4-server (default 32768)
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import AsyncIterator
from typing import Any

from .base import BaseEngine, GenerationOutput


class ExternalGGUFError(RuntimeError):
    pass


class ExternalGGUFEngine(BaseEngine):
    """BaseEngine implementation that proxies to a ds4-server subprocess."""

    def __init__(self, *, model_path: str, context_window: int = 32768,
                 host: str = "127.0.0.1", port: int = 11435):
        self._model_path = model_path
        self._context_window = context_window
        self._host = host
        self._port = port
        self._proc: asyncio.subprocess.Process | None = None
        self._base_url = f"http://{host}:{port}"
        self._started = False
        self._binary = os.environ.get("OMLX_DS4_SERVER_BIN") or shutil.which(
            "ds4-server"
        )
        if not self._binary:
            raise ExternalGGUFError(
                "External GGUF serving requires ds4-server on PATH or "
                "OMLX_DS4_SERVER_BIN (DwarfStar/antirez-ds4 build)"
            )

    @property
    def model_name(self) -> str:
        return self._model_path

    @property
    def model_type(self) -> str | None:
        return "external_gguf"

    def get_stats(self) -> dict[str, Any]:
        return {
            "engine": "external_gguf",
            "model": self._model_path,
            "binary": self._binary,
            "started": self._started,
        }

    def get_cache_stats(self) -> dict[str, Any] | None:
        return None

    @property
    def tokenizer(self) -> Any:
        # ds4-server owns tokenization; expose a minimal shim so engine-pool
        # tokenizer checks pass (encode is used only for stats).
        return _Ds4TokenizerShim()

    async def start(self) -> None:
        if self._started:
            return
        self._proc = await asyncio.create_subprocess_exec(
            self._binary,
            "-m", self._model_path,
            "--host", self._host,
            "--port", str(self._port),
            "--ctx", str(self._context_window),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        # wait for readiness
        for _ in range(60):
            if self._proc.returncode is not None:
                raise ExternalGGUFError(
                    f"ds4-server exited during startup: {self._proc.returncode}"
                )
            try:
                import urllib.request
                with urllib.request.urlopen(
                    f"{self._base_url}/health", timeout=2
                ):
                    self._started = True
                    return
            except Exception:
                await asyncio.sleep(0.5)
        raise ExternalGGUFError("ds4-server did not become ready")

    async def stop(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except TimeoutError:
                self._proc.kill()
                await self._proc.wait()
        self._proc = None
        self._started = False

    async def _post(self, path: str, payload: dict, timeout: float = 900):
        import urllib.request
        req = urllib.request.Request(
            f"{self._base_url}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: urllib.request.urlopen(req, timeout=timeout).read()
        )

    @staticmethod
    def _render_messages(messages: list[dict[str, Any]]) -> str:
        """Concatenate messages into ds4's single-prompt form."""
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content") or ""
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        await self.start()
        prompt = self._render_messages(messages)
        payload = {
            "model": "ds4",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        raw = await self._post("/v1/completions", payload)
        body = json.loads(raw.decode("utf-8"))
        choice = (body.get("choices") or [{}])[0]
        text = (choice.get("text") or "").strip()
        usage = body.get("usage") or {}
        return GenerationOutput(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=str(choice.get("finish_reason") or "stop"),
        )

    async def generate(
        self,
        prompt: str | list[int],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        await self.start()
        if isinstance(prompt, list):
            raise ExternalGGUFError(
                "External GGUF engine requires a text prompt, not token ids"
            )
        payload = {
            "model": "ds4",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop
        raw = await self._post("/v1/completions", payload)
        body = json.loads(raw.decode("utf-8"))
        choice = (body.get("choices") or [{}])[0]
        text = (choice.get("text") or "").strip()
        usage = body.get("usage") or {}
        return GenerationOutput(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=str(choice.get("finish_reason") or "stop"),
        )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        await self.start()
        prompt = self._render_messages(messages)
        payload = {
            "model": "ds4",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": True,
        }
        raw = await self._post("/v1/completions", payload)
        for line in raw.decode("utf-8").splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choice = (chunk.get("choices") or [{}])[0]
            text = (choice.get("delta") or choice).get("text", "")
            if text:
                yield GenerationOutput(text=text, new_text=text, finished=False)
        yield GenerationOutput(text="", new_text="", finished=True)

    async def stream_generate(
        self,
        prompt: str | list[int],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        await self.start()
        if isinstance(prompt, list):
            raise ExternalGGUFError(
                "External GGUF engine requires a text prompt, not token ids"
            )
        payload = {
            "model": "ds4",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }
        if stop:
            payload["stop"] = stop
        raw = await self._post("/v1/completions", payload)
        for line in raw.decode("utf-8").splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choice = (chunk.get("choices") or [{}])[0]
            text = (choice.get("delta") or choice).get("text", "")
            if text:
                yield GenerationOutput(text=text, new_text=text, finished=False)
        yield GenerationOutput(text="", new_text="", finished=True)


class _Ds4TokenizerShim:
    """Minimal tokenizer shim so engine-pool tokenizer checks pass."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    @property
    def eos_token_id(self) -> int | None:
        return None
