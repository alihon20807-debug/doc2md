"""High-speed multi-threaded downloader using Rust hf_transfer for fast Wi-Fi connections."""
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download

repo_id = sys.argv[1] if len(sys.argv) > 1 else "cyankiwi/gemma-4-12B-it-qat-AWQ-INT4"
cache_dir = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\aliho\.cache\huggingface\hub"

print(f"[FAST DOWNLOAD] Starting hf_transfer for: {repo_id}")
print(f"[CACHE PATH] {cache_dir}")
print(f"[START TIME] {time.strftime('%H:%M:%S')}")
print("=" * 65)

start = time.time()
try:
    path = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        max_workers=16,
        resume_download=True,
    )
    elapsed = time.time() - start
    print(f"\n[SUCCESS] Downloaded in {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)!")
    print(f"Local path: {path}")
except Exception as e:
    print(f"\n[ERROR] Download failed: {e}")
    sys.exit(1)
