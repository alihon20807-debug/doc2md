"""HTTP clients for VLM backends: vLLM (async), OpenRouter (async), and llama-server."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import time
from collections import deque
from typing import Any

import httpx
from PIL import Image

from doc2md.config import Settings
from doc2md.models import Bucket
from doc2md.prompts import SYSTEM_PROMPT

_MAX_RETRY_DELAY_S = 30.0


class VLMError(RuntimeError):
    pass


class _RateLimiter:
    """Blocks `acquire()` callers so no more than `per_minute` of them start
    within any trailing 60-second window. `per_minute <= 0` means unlimited.
    """

    def __init__(self, per_minute: float):
        self._per_minute = per_minute
        self._lock = asyncio.Lock()
        self._starts: deque[float] = deque()

    async def acquire(self) -> None:
        if self._per_minute <= 0:
            return
        async with self._lock:
            while True:
                now = time.monotonic()
                while self._starts and now - self._starts[0] >= 60.0:
                    self._starts.popleft()
                if len(self._starts) < self._per_minute:
                    self._starts.append(now)
                    return
                await asyncio.sleep(max(60.0 - (now - self._starts[0]), 0.01))


def _encode_image(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status in (500, 502, 503, 504, 429)
    return False


def _build_chat_payload(
    settings: Settings,
    image: Image.Image,
    prompt: str,
    system_prompt: str,
    *,
    model: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _encode_image(image)}},
                ],
            },
        ],
        "temperature": settings.vlm_temperature,
        "max_tokens": settings.vlm_max_tokens,
    }
    if model is not None:
        payload = {"model": model, **payload}
    return payload


async def _post_chat_completion_async(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_s: float,
    max_retries: int,
) -> str:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=timeout_s)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < max_retries and _is_transient_error(exc):
                delay = min(1.0 * (2 ** attempt), _MAX_RETRY_DELAY_S)
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
                    retry_after = exc.response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = max(float(retry_after), delay)
                        except ValueError:
                            pass
                await asyncio.sleep(delay)
            else:
                break
    raise VLMError(f"VLM request to {url} failed after retries: {last_error}") from last_error


class AsyncVLLMClient:
    """Talks asynchronously to a local vLLM server via OpenAI-compatible /v1/chat/completions."""

    def __init__(self, settings: Settings):
        self._settings = settings
        base_url = settings.vllm_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        self._base_url = base_url
        self._client = httpx.AsyncClient(timeout=settings.vlm_timeout_s)

    async def close(self) -> None:
        await self._client.aclose()

    async def health_check(self) -> None:
        url = f"{self._base_url}/models"
        try:
            response = await self._client.get(url, timeout=5.0)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise VLMError(
                f"Cannot reach vLLM server at {self._base_url} "
                f"(is vLLM running with OpenAI API server enabled?): {exc}"
            ) from exc

    async def ask(
        self, image: Image.Image, prompt: str, system_prompt: str = SYSTEM_PROMPT, *, bucket: Bucket | None = None
    ) -> str:
        settings = self._settings
        payload = _build_chat_payload(settings, image, prompt, system_prompt, model=settings.vllm_model)
        url = f"{self._base_url}/chat/completions"
        return await _post_chat_completion_async(
            self._client, url, {}, payload, settings.vlm_timeout_s, settings.vlm_max_retries
        )


class AsyncOpenRouterClient:
    """Talks to OpenRouter's hosted API asynchronously."""

    def __init__(self, settings: Settings):
        self._settings = settings
        api_key = settings.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise VLMError(
                "OpenRouter backend selected but no API key found - pass --openrouter-api-key "
                "or set the OPENROUTER_API_KEY environment variable."
            )
        self._api_key = api_key
        self._client = httpx.AsyncClient(timeout=settings.vlm_timeout_s)
        self._rate_limiter = _RateLimiter(settings.vlm_requests_per_minute)

    async def close(self) -> None:
        await self._client.aclose()

    async def health_check(self) -> None:
        url = f"{self._settings.openrouter_base_url.rstrip('/')}/models"
        try:
            response = await self._client.get(url, headers=self._headers(), timeout=10.0)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise VLMError(f"Cannot reach OpenRouter at {url} (bad API key or network issue?): {exc}") from exc

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": "https://github.com/doc2md/doc2md",
            "X-Title": "doc2md",
        }

    def _model_for_bucket(self, bucket: Bucket | None) -> str:
        if bucket is Bucket.TEXT_LIKE:
            return self._settings.openrouter_text_model
        return self._settings.openrouter_diagram_model

    async def ask(
        self, image: Image.Image, prompt: str, system_prompt: str = SYSTEM_PROMPT, *, bucket: Bucket | None = None
    ) -> str:
        settings = self._settings
        model = self._model_for_bucket(bucket)
        payload = _build_chat_payload(settings, image, prompt, system_prompt, model=model)
        url = f"{settings.openrouter_base_url.rstrip('/')}/chat/completions"
        await self._rate_limiter.acquire()
        return await _post_chat_completion_async(
            self._client, url, self._headers(), payload, settings.vlm_timeout_s, settings.vlm_max_retries
        )


class VLMClient:
    """Legacy llama-server client - natively async, same interface shape as
    AsyncVLLMClient/AsyncOpenRouterClient."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = httpx.AsyncClient(timeout=settings.vlm_timeout_s)

    async def close(self) -> None:
        await self._client.aclose()

    async def health_check(self) -> None:
        url = f"{self._settings.llama_server_url.rstrip('/')}/health"
        try:
            response = await self._client.get(url, timeout=5.0)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise VLMError(
                f"Cannot reach llama-server at {self._settings.llama_server_url}: {exc}"
            ) from exc

    async def ask(
        self, image: Image.Image, prompt: str, system_prompt: str = SYSTEM_PROMPT, *, bucket: Bucket | None = None
    ) -> str:
        settings = self._settings
        payload = _build_chat_payload(settings, image, prompt, system_prompt)
        url = f"{settings.llama_server_url.rstrip('/')}/v1/chat/completions"
        return await _post_chat_completion_async(
            self._client, url, {}, payload, settings.vlm_timeout_s, settings.vlm_max_retries
        )


def create_vlm_client(settings: Settings) -> AsyncVLLMClient | AsyncOpenRouterClient | VLMClient:
    if settings.vlm_backend == "openrouter":
        return AsyncOpenRouterClient(settings)
    if settings.vlm_backend == "llama_server":
        return VLMClient(settings)
    return AsyncVLLMClient(settings)

