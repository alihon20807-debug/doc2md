"""Concurrent max-throughput test against a running vLLM server, using a
realistic doc2md-style image+prompt request (real page-region crop, real
system+text-transcription prompt from doc2md/prompts.py) rather than a
synthetic probe.

This is NOT a doc2md pipeline test - it talks to the OpenAI-compatible
/v1/chat/completions endpoint directly with httpx, bypassing doc2md
entirely, so it can be pointed at any --vllm-model/--vllm-url combination
to characterize server throughput in isolation.

Run under the same Python environment vLLM is installed in (it needs httpx
and PIL). Under WSL2, that's typically NOT the system python3 - use the
same interpreter as `vllm serve`, e.g.:

    /home/aliho/.python312/bin/python3.12 scripts/vllm_throughput_test.py

Concurrency and duration are configurable via env vars so you can sweep
--max-num-seqs values without editing the file:

    TEST_CONCURRENCY=128 TEST_DURATION=30 python3.12 scripts/vllm_throughput_test.py

Reads docs/vllm_throughput_test_region.png as the test image (a real
sample_docs/test.pdf page-region crop, checked into the repo so this script
doesn't depend on any ephemeral scratch directory). If that path doesn't
resolve from wherever you're running this (e.g. a different WSL distro),
override IMAGE_PATH below or pass an absolute path.
"""

import asyncio
import base64
import io
import os
import time

import httpx
from PIL import Image

MODEL = "cyankiwi/gemma-4-12B-it-qat-AWQ-INT4"
URL = "http://127.0.0.1:8000/v1/chat/completions"
IMAGE_PATH = os.environ.get(
    "TEST_IMAGE_PATH",
    "/mnt/c/Users/aliho/Documents/Codes/doc2md/docs/vllm_throughput_test_region.png",
)

# Mirrors doc2md/prompts.py's SYSTEM_PROMPT and text_prompt() default -
# duplicated rather than imported so this script has no dependency on the
# doc2md package or its venv, only on whatever env vLLM itself runs under.
SYSTEM_PROMPT = (
    "You are a meticulous OCR and document-transcription assistant. You are given a single "
    "cropped region from a scanned document page. Respond with ONLY the requested Markdown "
    "content for that region - no preamble, no explanations, no surrounding commentary."
)
TEXT_PROMPT = (
    "Transcribe the text in this image exactly as written, preserving reading order, line "
    "breaks between distinct items, and any obvious list/bullet structure as Markdown. Do not "
    "translate, summarize, or add anything not present in the image. Output only the "
    "transcribed Markdown text. If the region is blank or contains no legible text, output "
    "nothing."
)

CONCURRENCY = int(os.environ.get("TEST_CONCURRENCY", 16))
DURATION_S = int(os.environ.get("TEST_DURATION", 30))
MAX_TOKENS = 300


def load_image_data_url(path: str) -> str:
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


async def one_request(client: httpx.AsyncClient, image_url: str) -> tuple[int, float]:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": TEXT_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        "temperature": 0.1,
        "max_tokens": MAX_TOKENS,
    }
    t0 = time.perf_counter()
    resp = await client.post(URL, json=payload, timeout=120.0)
    resp.raise_for_status()
    data = resp.json()
    elapsed = time.perf_counter() - t0
    completion_tokens = data["usage"]["completion_tokens"]
    return completion_tokens, elapsed


async def worker(worker_id, client, image_url, stop_at, results):
    while time.perf_counter() < stop_at:
        try:
            tokens, elapsed = await one_request(client, image_url)
            results.append((tokens, elapsed))
        except Exception as exc:  # noqa: BLE001
            results.append((None, str(exc)))


async def main():
    image_url = load_image_data_url(IMAGE_PATH)
    limits = httpx.Limits(max_connections=CONCURRENCY + 4, max_keepalive_connections=CONCURRENCY)
    async with httpx.AsyncClient(limits=limits) as client:
        # warmup: one request to make sure CUDA graphs / caches are warm before timing
        print("warmup request...")
        wt, we = await one_request(client, image_url)
        print(f"warmup ok: {wt} tokens in {we:.2f}s")

        results: list[tuple] = []
        start = time.perf_counter()
        stop_at = start + DURATION_S
        tasks = [
            asyncio.create_task(worker(i, client, image_url, stop_at, results))
            for i in range(CONCURRENCY)
        ]
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - start

    ok = [(t, e) for t, e in results if isinstance(t, int)]
    failed = [r for r in results if not isinstance(r[0], int)]
    total_tokens = sum(t for t, _ in ok)
    n = len(ok)
    print()
    print(f"concurrency={CONCURRENCY}  duration={wall:.1f}s")
    print(f"completed requests: {n}  failed: {len(failed)}")
    if failed:
        print("sample failure:", failed[0][1][:300])
    if n:
        avg_tokens = total_tokens / n
        avg_latency = sum(e for _, e in ok) / n
        print(f"avg completion_tokens/request: {avg_tokens:.1f}")
        print(f"avg latency/request: {avg_latency:.2f}s")
        print(f"TOTAL COMPLETION TOKENS: {total_tokens}")
        print(f"AGGREGATE THROUGHPUT: {total_tokens / wall:.1f} tokens/sec")


if __name__ == "__main__":
    asyncio.run(main())
