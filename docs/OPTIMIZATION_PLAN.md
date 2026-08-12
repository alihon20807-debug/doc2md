# doc2md Optimization Plan (2026-08-12)

Scope: make the vLLM/Gemma-4 conversion pipeline faster and cheaper without
losing accuracy. Everything below is either (a) already done and measured
this session, (b) investigated and found NOT safe to do (kept here so it
isn't re-attempted from scratch), or (c) a scoped, evidence-backed next
step not yet implemented.

## Done this session (recap)

1. **Server concurrency**: `--max-num-seqs 128` re-confirmed better than 96
   (1486.8 vs 1216.4 tok/s aggregate, no queueing either way, fresh
   apples-to-apples rerun). No change - already configured.
2. **Client concurrency**: doc2md's own `--concurrency` (32/64/128) tested
   against the real 508-region PPTX - all three within noise of each other
   (112-116s extraction). Default of 32 confirmed sufficient; the
   bottleneck was never client-side fan-out.
3. **Thinking mode**: confirmed OFF by inspecting the model's own chat
   template directly (not assumed) - `enable_thinking` defaults false, and
   the template actively injects an empty, pre-closed thought channel when
   it's off. Nothing to change.
4. **Per-bucket `max_tokens`**: added `vlm_max_tokens_text_like` (768,
   `config.py`) separate from `vlm_max_tokens` (2048, kept for
   `TABLE`/`PICTURE`). Real regions rarely need more than ~150-200 tokens; a
   tighter cap makes an actual decoding loop hit the ceiling (and get
   caught by the existing degenerate-retry check) much sooner. Measured:
   extraction time 112-116s -> **92s** on the same PPTX, zero decoding-loop
   notes, *more* total output than the pre-fix baseline (nothing legitimate
   was cut short).

## Investigated and found NOT safe: filtering "tiny" regions

The original hypothesis for this investigation (small/few-pixel image
regions being false-positive noise wasting VLM calls) does not hold up
under verification, and the obvious fix was **not** implemented as a
result. Recorded in detail so a future session doesn't re-propose the same
thing without re-deriving why it's wrong:

- 53/508 (10.4%) of the real PPTX's detected regions have bbox area
  <1000px², with lower average detection confidence (median 0.53) than the
  overall population (median 0.65) - this part of the original hypothesis
  is true.
- The first 3 samples visually inspected (crops literally read as images)
  were "86", "IP", "MS" - looked exactly like noise (a stray page number, a
  watermark fragment) at first glance.
- **But then a 4th, slightly larger sample (644px², "Trailer (optional)")
  turned out to be real, meaningful body content** - one bullet line from
  an actual slide about network-packet encapsulation (Header/Payload/
  Trailer terminology).
- Full region dump for that specific page (24) showed why: mineru detects
  **one region per line**, not one region per paragraph. A paragraph with
  short lines ("Trailer", "Header" used as one-word sub-labels within a
  bulleted breakdown) produces small, low-confidence-looking boxes for
  entirely mundane reasons - short text is small - not because anything is
  wrong with the detection.
- Re-checking the original "86" sample's *position* (not just its size)
  confirmed it sits mid-page-height in the same left-margin column as every
  other line on that page, not in a footer/page-number band - it's coincidence
  that its transcribed content happened to equal the page number, not evidence
  it *is* a page-number artifact.
- **Conclusion: small bbox area does not reliably predict "not real
  content" for this document.** An area or confidence-based auto-drop
  filter would have silently deleted real lecture content (at minimum
  "Trailer (optional)", plausibly others in the 53-region set never
  individually inspected). Not implemented. If revisited, it would need a
  much stronger signal than geometry alone (e.g., OCR'd content matching a
  known page-number/footer pattern, or agreement with the layout model's
  own *native* `page_number`/`header`/`footer` label class rather than
  guessing from box size) - and even then, spot-check real samples before
  shipping, the way this investigation did, because a plausible-looking
  wrong filter is worse than no filter.

## Line-region merging: implemented, works, shipped opt-in (not default) - 2026-08-12

Implemented as sketched below (`doc2md/layout_engines/merge.py`,
`Settings.region_merge_enabled`/`region_merge_max_gap_frac`/
`region_merge_x_align_px`). Measured on the real 508-region PPTX:

- **Region count: 508 -> 385 (24.2% fewer VLM calls)**, applied uniformly
  before crop/VLM (wired into `finalize_regions()` for 5 engines, and
  directly into `mineru_engine.py`, which bypasses `finalize_regions`).
- **Wall time: 102.5s -> 81.6s** (full pipeline, live server, `--concurrency 128`).

**Not enabled by default, shipped as opt-in (`region_merge_enabled=False`).**
While verifying content preservation (comparing merge-off vs merge-on
output word-for-word), a real, previously undocumented bug surfaced that
made a clean before/after comparison impossible in the time available:

**Newly discovered, pre-existing bug (NOT caused by the merge feature -
confirmed present in the unmerged baseline too): some slides in this real
PPTX have overlapping/near-duplicate detected regions that survive
`suppress_contained_duplicates()` because they overlap without one fully
containing the other (the dedup check requires ~80% containment). Each
overlapping region gets its own separate VLM call, and each independently
re-transcribes the same underlying content with different paraphrasing -
producing visibly duplicated, reworded bullet points in the output (e.g.
"Explain why abstraction and layering..." appears 2-4 times with slightly
different wording). On at least one slide, this also produced a clear
hallucination: a fabricated "TASK: CONFIDENTIALITY NOTICE" corporate
boilerplate paragraph that does not appear anywhere in the source
document - confirmed by rendering the actual slide images directly and
visually inspecting them.** This is the same "fluent-but-wrong VLM
response, not repetition-shaped, so not caught by `_looks_degenerate()`"
residual risk already flagged in `CLAUDE.md`'s decoding-loop-fix entry,
now with a concrete reproduction. It was present in *both* the merge-off
and merge-on runs (with different exact wording each time, since VLM
generation isn't perfectly deterministic even at `temperature=0.1`), which
is what proved it's unrelated to the merge feature - but it also means the
word-level diff between merge-off/merge-on output was comparing two
different flavors of the *same* pre-existing hallucination against each
other on that slide, contaminating the comparison and leaving genuine
uncertainty about whether merging is content-safe on the rest of the
document. Given the time available, the responsible call was to ship the
merge feature as opt-in rather than assert it's fully safe without a clean
signal - flip `region_merge_enabled=True` once this overlap/hallucination
bug is fixed and a clean comparison becomes possible.

## Root cause found and fixed: full-page-fallback hallucination (2026-08-12, follow-up)

The "CONFIDENTIALITY NOTICE" hallucination above was root-caused by testing
every text region on pages 1-10 individually against the live VLM and
checking which one's output contained "confidential" - found in one call:
`page=4 bbox=(0, 0, 2666, 1500) score=1.00`. That bbox and score signature
means MinerU detected **zero** real regions on page 4, triggering
`full_page_fallback_region()`'s synthesized full-page region - and page 4's
real content turned out to be a small ~228x188px cluster of tiny text in
the top-left corner of an otherwise blank 2666x1500 canvas (confirmed by
rendering the page directly). The VLM, given a giant mostly-blank image
with the real content reduced to a barely-legible sliver after whatever
downscaling its vision preprocessing applies, fabricated plausible generic
corporate-document boilerplate instead of saying it couldn't read
anything - the same "fluent-but-wrong, not repetition-shaped" residual risk
`CLAUDE.md`'s decoding-loop-fix entry already named, now with a concrete
reproduction.

**Scope check**: this is not a one-off. `MinerULayoutEngine` returns zero
regions (triggering the full-page fallback) on **15 of 98 pages (15.3%)**
of this real PPTX, and every one of them has the same signature - a small
content cluster on an otherwise blank page (this presentation's template
consistently places title/duration/page-number-style content in the
top-left corner with a lot of surrounding whitespace). That's a
much bigger blast radius than "one bad slide."

**Fix**: `layout_engines/base.py`'s new `_content_bbox()` finds the actual
non-blank ink on the page (grayscale -> invert -> threshold -> PIL
`getbbox()`) and `full_page_fallback_region()` now uses that (padded by
`crop_padding_px`) instead of always the literal full page - falling back
to the full page only if no content is found (genuinely blank) or the
content already spans ~90%+ of the page (no point tightening further).
Verified across all 15 fallback pages: bboxes shrank to 0.2%-10.3% of the
page area, all starting near the same top-left corner, consistent with the
template. Lowering `layout_confidence` globally was tried first and
rejected - it's not a clean fix (tested 0.45/0.3/0.2/0.1/0.05 on page 4:
inconsistent, non-monotonic region counts, and 0.05 introduced spurious
junk detections) - the content-bbox approach doesn't touch detection
confidence at all, so it has no such fragility.

**Verified end-to-end**: full 508-region PPTX conversion after the fix -
zero "CONFIDENTIAL" mentions anywhere in the output (down from at least
one confirmed hallucination before), zero decoding-loop notes, wall time
unchanged (~92s, this fix doesn't touch region count, only fallback bbox
sizing).

## Region merging, re-evaluated after the fallback fix

Re-ran the merge-on/merge-off content comparison now that the
fallback-hallucination bug (the actual contaminating factor before) is
fixed. Clean this time in the sense that **zero hallucination** appeared in
either run - but a new, real signal appeared instead: merge-on produced
**~17-19% fewer total words** than merge-off (8544 vs 10267 on one run),
with the "missing" words skewing toward real content terms (`protocol`,
`layer`, `message`, `application`), not just filler. Interpretation: this
is a VLM *transcription* pipeline, not literal OCR - text_prompt() asks the
model not to summarize, but a denser multi-line merged crop appears to give
it more room to paraphrase/condense than a single, unambiguous one-line
crop does. Some "extra" words also appeared (`rightarrow`, `square`,
diagram-syntax-looking tokens - Mermaid/SVG leakage on a picture-adjacent
region), a smaller but separate signal worth a future look.

**Decision: `region_merge_enabled` stays `False` by default.** Not because
of hallucination risk (that's resolved and was never actually caused by
merging - see above), but because there's now direct evidence of a real,
if modest, transcription-completeness tradeoff that a user should opt into
knowingly, not get by default. The speed win is real and unchanged (508 ->
385 regions, ~100s -> ~81s wall time) for anyone who decides the
completeness tradeoff is acceptable for their use case.

## Recommended next step (superseded by the above - kept for history): merge same-column adjacent line regions

The real, safe opportunity this investigation surfaced isn't "drop small
regions," it's "don't pay a full separate VLM round-trip for every single
short line." Evidence: page 24 alone has **8 separate detected regions**,
several of them single words/short phrases ("Trailer (optional)", "Header")
that are really just consecutive lines of one bulleted paragraph. Each
region pays the same fixed overhead regardless of how little text it holds
- a full network round-trip, prefill of the system prompt + user prompt
tokens, and a full crop/encode cycle - so a paragraph mineru splits into 5
line-regions costs ~5x the fixed overhead of 1 region with all the same
content, for zero accuracy benefit (unlike the drop-based filter above,
this loses no information at all - it only changes how many VLM calls the
same total content is packaged into).

**Sketch (not implemented yet - flagged as the top priority for the next
work session):**

- Add a merge pass, most naturally alongside `finalize_regions()` in
  `layout_engines/base.py` (used by 5 of 6 engines) plus a MinerU-specific
  equivalent (MinerU skips `finalize_regions()` - it has its own
  reading-order head, per `mineru_engine.py`).
- Merge candidates: consecutive regions (by reading order) with the *same*
  `bucket` (only ever `TEXT_LIKE` - never merge across `TABLE`/`PICTURE`,
  which need their own dedicated prompts) and the *same* raw `label`
  (e.g. don't merge a `paragraph_title` into a `text` line), where the
  vertical gap between them is small relative to their own height and their
  horizontal (x) ranges overlap or nearly align - i.e. genuinely stacked
  lines of one visual block, not two unrelated regions that happen to be
  reading-order-adjacent.
- Merged region's bbox = the union of the merged boxes; its crop is one
  crop of that union, sent as a single VLM call with the existing
  `text_prompt()` (multi-line input is already normal for that prompt).
- **Safety check before shipping**: verify on the real 508-region PPTX (and
  ideally the real multi-page PDF too) that (a) region count drops
  meaningfully, (b) wall time drops correspondingly, and (c) the resulting
  markdown's real content is a superset match of the pre-merge output line
  by line (nothing merged away, nothing duplicated) - not just "it looks
  plausible," given this session's tiny-region investigation is a direct
  lesson in why that's not enough.
- Expected impact: if even half of the 53 small-area regions found above
  are line-fragments of larger paragraphs (plausible, given the one sampled
  in depth was), that's a meaningful cut in total VLM call count on a
  real-world multi-line-heavy document like this one, on top of the
  already-measured max_tokens win.

## Other candidate optimizations (lower priority / not evidenced yet)

- **OCR fast path** (`--ocr-fast-path`, off by default): could offload
  plain `TEXT_LIKE` regions to mineru's/paddleocr's/docling's bundled OCR
  instead of the VLM, which would be much cheaper per region. Not enabled
  by default because OCR-vs-VLM transcription quality has never been
  head-to-head compared on a real document in this repo - would need that
  comparison before recommending it as a default, not just as an opt-in.
- **Crop resolution capping**: no max-dimension cap exists in `crop.py` -
  crops (including full-page fallback regions) go out at the full render
  DPI (200) resolution. Whether this matters is unmeasured - Gemma 4's
  encoder-free tokenization is already fairly cheap (~167 vision tokens for
  an 862x892 real crop, per earlier measurement in this repo), so a large
  crop may not cost proportionally more than the vision-token math implies.
  Needs actual measurement (compare vision token counts across a range of
  real crop sizes) before deciding this is worth doing.
- **System prompt trimming**: every request pays the same fixed system
  prompt prefill cost regardless of region size; a shorter system prompt
  would save a small, fixed amount multiplied by hundreds of regions/doc.
  Low individual impact, not investigated this session.
- **Real-world request-latency variance** (open since the previous
  concurrency investigation): a chunk of real regions show 30-46s latency
  for ~150-200 tokens of clean, non-looping, non-retried output - not
  explained by anything found so far (not thinking, not HTTP retries, not
  the degenerate-retry path). Needs deeper profiling (e.g. correlate
  per-request elapsed time against crop pixel dimensions / vision token
  count) to even form a real hypothesis. Unsolved.
- **Cross-engine layout benchmark**: still doesn't exist (see
  `docs/ARCHITECTURE.md` §7) - only `mineru` has real-world verification.
  Not an optimization exactly, but blocks evaluating whether a different
  default engine would be faster/more accurate.

## Explicit non-goals for this plan

- Fine-tuning Gemma 4 locally - deferred by explicit user request, tracked
  separately, not part of this optimization pass.
- Multi-phase describe-then-transcribe prompting - a real design change
  (see `CLAUDE.md`'s "Prompt architecture" note), not an "optimization" in
  the speed/cost sense discussed here; roughly doubles cost per region.
- Any region-dropping filter based on size/confidence alone - see the
  "investigated and found NOT safe" section above. Don't re-propose this
  without a stronger, content-aware signal than geometry.
