# Keye DSA Trace Pilot

These scripts reproduce the BFCL multi-turn score/top-k pilot used by the
`codex/keye-sm80-support` branch.

## Trace configuration

Start the Keye server with the SM80 path and these trace variables:

```bash
export KEYE_SM80_DSA=1
export KEYE_SM80_TRACE_DIR=/absolute/path/to/run/events
export KEYE_SM80_TRACE_LAYERS=0,1,15,16,31,32,46,47
export KEYE_SM80_TRACE_MODE=both
export KEYE_SM80_TRACE_DECODE_STEPS=32
export KEYE_SM80_TRACE_CHUNK_STEPS=32
export KEYE_SM80_TRACE_RID_PREFIX=bfclseg__
```

`KEYE_SM80_TRACE_DECODE_STEPS` is enforced independently for every
`(request ID, layer)` pair. Schema v3 event files keep the FP32 score vector,
int32 top-k indices, valid score length, request ID, layer ID, and decode step
in one record.

With `KEYE_SM80_TRACE_CHUNK_STEPS` enabled, schema v4 stores one atomic file
per `(request ID, layer)` chunk. This avoids creating one small file per decode
step while retaining FP32 scores and int32 top-k indices. The optional request
prefix filter excludes health checks and warm-up requests.

## Replay

`run_bfcl_teacher_forced.py` selects one BFCL v4 `multi_turn_base` case and one
`multi_turn_long_context` case. It uses greedy single-request inference and
replaces every sampled answer with the dataset ground-truth action plus a
deterministic observation before the next round.

```bash
.venv/bin/python scripts/keye_trace/run_bfcl_teacher_forced.py \
  --bfcl-root data/external/gorilla/berkeley-function-call-leaderboard \
  --output-dir data/agent-score-trace/<run-id>
```

`run_bfcl_segmented.py` is the full semantic-context experiment. It
deterministically selects stratified BFCL trajectories, executes the official
BFCL ground-truth tool calls, renders the exact Keye chat prompt, labels every
prompt token by semantic segment, and sends the resulting token IDs directly
to SGLang:

```bash
.venv/bin/python scripts/keye_trace/run_bfcl_segmented.py \
  --bfcl-root data/external/gorilla/berkeley-function-call-leaderboard \
  --output-dir data/agent-score-trace/<run-id> \
  --prepare-only

.venv/bin/python scripts/keye_trace/run_bfcl_segmented.py \
  --bfcl-root data/external/gorilla/berkeley-function-call-leaderboard \
  --output-dir data/agent-score-trace/<run-id> \
  --reuse-prepared
```

## Analysis

`analyze_score_trace.py` validates the full `9 requests × 8 layers × 32 steps`
grid, writes typed Parquet tables, and exports PDF plus 300 dpi PNG figures.
Every step comparison is grouped by layer; no cross-layer pooling is used for
step stability or adjacent-step score-change metrics.

```bash
.venv/bin/python scripts/keye_trace/analyze_score_trace.py \
  --run-dir data/agent-score-trace/<run-id> \
  --style /path/to/matplotlib_style.mplstyle
```

The analysis requires `pandas`, `pyarrow`, `scipy`, and `matplotlib`. Raw traces
and generated figures remain under the Git-ignored `data/` directory.
