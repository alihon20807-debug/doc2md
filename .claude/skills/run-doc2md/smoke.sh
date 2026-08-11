#!/usr/bin/env bash
# Smoke-test driver for doc2md: verifies the environment is set up, then
# runs a real conversion against sample_docs/test.pdf and checks the output
# actually has the structure doc2md is supposed to produce (a heading, a
# real Markdown table, a Mermaid block) - not just that the command exited 0.
#
# Run from the repo root: bash .claude/skills/run-doc2md/smoke.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

pass() { echo "[PASS] $1"; }
fail() { echo "[FAIL] $1"; FAILED=1; }
FAILED=0

echo "== 1. CLI is installed and importable =="
if uv run doc2md --help >/tmp/doc2md_help.out 2>&1; then
  pass "uv run doc2md --help exits 0"
else
  fail "uv run doc2md --help failed - run 'uv sync' first (see SKILL.md Setup)"
  cat /tmp/doc2md_help.out
  exit 1
fi

echo "== 2. VLM backend server reachability (default: vLLM at :8000) =="
if curl -s -m 5 http://127.0.0.1:8000/v1/models >/tmp/doc2md_models.out 2>&1; then
  pass "vLLM server reachable at http://127.0.0.1:8000"
  SERVER_UP=1
else
  echo "[SKIP] no vLLM server reachable at http://127.0.0.1:8000 - the real"
  echo "       conversion test below needs one running (see SKILL.md"
  echo "       Prerequisites). Layout-detection-only checks still run."
  SERVER_UP=0
fi

echo "== 3. Layout detection runs standalone (no VLM server needed) =="
if uv run python -c "
from doc2md.render import render_input
from doc2md.config import Settings
from doc2md.layout_engines import get_layout_engine
from pathlib import Path
pages = render_input(Path('sample_docs/test.pdf'), dpi=150)
regions = get_layout_engine(Settings(layout_engine='mineru')).detect(pages[0])
assert len(regions) > 0, 'expected at least one detected region'
print(f'detected {len(regions)} regions: {[r.label for r in regions]}')
" 2>/tmp/doc2md_layout.err; then
  pass "mineru layout engine detects real regions in sample_docs/test.pdf"
else
  fail "layout detection failed"
  cat /tmp/doc2md_layout.err
fi

if [ "$SERVER_UP" = "1" ]; then
  echo "== 4. Real end-to-end conversion against the live VLM server =="
  OUT_DIR="$(mktemp -d)"
  if uv run doc2md sample_docs/test.pdf -o "$OUT_DIR" >/tmp/doc2md_convert.out 2>&1; then
    pass "uv run doc2md sample_docs/test.pdf -o <dir> exits 0"
  else
    fail "conversion command failed"
    cat /tmp/doc2md_convert.out
  fi

  MD_FILE="$OUT_DIR/test.md"
  if [ -f "$MD_FILE" ]; then
    pass "output file $MD_FILE was created"
    grep -qE '^#' "$MD_FILE" && pass "output contains a Markdown heading" || fail "no Markdown heading found (title region wasn't transcribed as a heading)"
    grep -qE '^\|.*\|' "$MD_FILE" && pass "output contains a Markdown table row" || fail "no Markdown table found (table region wasn't transcribed correctly)"
    grep -q '```mermaid' "$MD_FILE" && pass "output contains a Mermaid diagram block" || fail "no Mermaid block found (diagram wasn't redrawn as Mermaid)"
  else
    fail "expected output file $MD_FILE does not exist"
  fi
  rm -rf "$OUT_DIR"
else
  echo "== 4. SKIPPED (no VLM server reachable) =="
fi

echo "=================================="
if [ "$FAILED" = "0" ]; then
  echo "SMOKE TEST: ALL CHECKS PASSED"
  exit 0
else
  echo "SMOKE TEST: ONE OR MORE CHECKS FAILED"
  exit 1
fi
