# Setting up the local VLM (Gemma 4 12B via llama.cpp)

doc2md talks to a running `llama-server` instance over its OpenAI-compatible
`/v1/chat/completions` endpoint. This document is how that server was set up on
this machine (RTX 5080 Laptop GPU, 16 GB VRAM, Blackwell/sm_120).

## 1. Get `llama-server`

Blackwell (sm_120) needs a llama.cpp build against CUDA 12.8+. The official
Windows CUDA-13.3 release build is used here (CUDA 13.x ships native Blackwell
codegen). Download from the `ggml-org/llama.cpp` GitHub releases page:

- `llama-<tag>-bin-win-cuda-13.3-x64.zip` — the server/CLI binaries
- `cudart-llama-bin-win-cuda-13.3-x64.zip` — matching CUDA runtime DLLs

Extract both zips into the same directory (e.g. `llama.cpp-bin/`, which is
git-ignored in this repo) so `llama-server.exe` sits next to the `cudart*.dll`
files it needs.

If a GPU isn't detected with the CUDA-13.3 build, fall back to the
`win-cuda-12.4-x64.zip` build (older toolkit, relies on PTX JIT to run on
Blackwell — works, just slower to start). Only build llama.cpp from source
(needs CUDA Toolkit 12.8+ and MSVC Build Tools) if both prebuilt options fail.

## 2. Get the model

Official quantized GGUFs: [`ggml-org/gemma-4-12B-it-GGUF`](https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF).

Default: **Q4_0** (~7.2 GB + ~0.2 GB mmproj) — leaves plenty of the 16 GB VRAM
budget free for KV cache and image tokens. **Q8_0** (~12.7 GB) is available
for higher quality if you don't mind a tighter VRAM margin.

```
gemma-4-12B-it-Q4_0.gguf
mmproj-gemma-4-12B-it-Q8_0.gguf   (mmproj is small; Q8_0 mmproj is fine even with a Q4_0 main model)
```

## 3. Launch the server

From the directory containing `llama-server.exe`:

```
.\llama-server.exe ^
  --model <path>\gemma-4-12B-it-Q4_0.gguf ^
  --mmproj <path>\mmproj-gemma-4-12B-it-Q8_0.gguf ^
  -ngl 999 ^
  --ctx-size 8192 ^
  --host 127.0.0.1 --port 8080
```

`-ngl 999` offloads all layers to the GPU. Increase `--ctx-size` if pages
produce many regions (each region is one request, so this mostly matters for
single very content-dense crops).

## 4. Verify

```
curl http://127.0.0.1:8080/health
```

should return `{"status":"ok"}`. Then check GPU usage with `nvidia-smi` while
running a multimodal request (see the project README for a curl example) to
confirm inference is actually happening on the GPU and not falling back to CPU.
