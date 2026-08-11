# PowerShell Launch Script for llama.cpp on RTX 5080 Laptop
$ErrorActionPreference = "Stop"

# Process Cleanup: Terminate any leftover instances holding VRAM
Get-Process -Name "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force

Write-Host "[INIT] Launching llama-server with optimized parameters..." -ForegroundColor Green

$MODEL_PATH = "models/gemma-4-12B-it-Q4_0.gguf"
$LLAMA_BIN = ".\llama.cpp-bin\llama-server.exe"

& $LLAMA_BIN `
  -m $MODEL_PATH `
  -ngl 99 `
  -fa on `
  -t 8 `
  -tb 16 `
  -np 1 `
  -ub 1024 `
  -c 4096 `
  --host 127.0.0.1 `
  --port 8080
