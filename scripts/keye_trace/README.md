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

The replay scripts always persist the full SGLang response in
`requests.jsonl`, including generated text, `output_ids`, finish reason, prompt
and completion token counts, and latency. They respect EOS by default so saved
responses remain valid inference outputs. Use `--ignore-eos` only when a fixed
decode length is explicitly more important than response quality.

With `KEYE_SM80_TRACE_CHUNK_STEPS` enabled, schema v4 stores one atomic file
per `(request ID, layer)` chunk. This avoids creating one small file per decode
step while retaining FP32 scores and int32 top-k indices. The optional request
prefix filter excludes health checks and warm-up requests.

For 32K--128K collection, use compact schema v5 instead of full score vectors:

```bash
export KEYE_SM80_TRACE_MODE=compact
export KEYE_SM80_TRACE_COMPACT_K=4096
export KEYE_SM80_TRACE_COMPACT_RANKS=2048,2560,3072,4096
export KEYE_SM80_TRACE_SCORE_BLOCK_SIZE=256
export KEYE_SM80_TRACE_FULL_SCORE_LAYERS=0,23,47
export KEYE_SM80_TRACE_FULL_SCORE_RID_PREFIXES=longtrace__shadow__
export KEYE_SM80_EXACT_TOPK=1
```

`KEYE_SM80_EXACT_TOPK=1` makes `torch.topk` the serving and trace reference.
Use it for correctness experiments: `fast_topk_v2` uses a bounded shared-memory
radix path and can replace more than a few boundary entries when real DSA scores
are highly clustered. Schema v5 records the selected backend in `topk_backend`.

Schema v5 keeps the serving kernel's canonical top-2048 and a configurable
sorted candidate prefix (top-4096 by default; top-8192 for the large-candidate
gate), including indices and FP32 scores, rank thresholds, and per-256-token-block score
moments plus threshold counts. Full FP32 score vectors are retained only when
both the optional layer and request-prefix filters match. Page bitmaps and
segment labels are derived offline from candidate indices and the prepared
request sidecars, avoiding duplicate data in every layer chunk. Compact mode
requires chunked decode tracing.

Validate every compact chunk, including any retained full-score shadows, with:

```bash
.venv/bin/python scripts/keye_trace/validate_compact_trace_v5.py \
  --run-dir data/compact-trace-v5/<run-id>
```

Audit an earlier fast-kernel run against the exact top-4096 saved by v5:

```bash
.venv/bin/python scripts/keye_trace/analyze_fast_topk_correctness.py \
  --run-dir data/compact-trace-v5/<run-id>
```

On the current mixed-GPU host, select the L40S and A6000 by UUID rather than
`CUDA_VISIBLE_DEVICES=0,2`: CUDA ordinal 2 resolves to the RTX 3090 here even
though `nvidia-smi` labels the A6000 as index 2.

## GLM-5.1 DSA motivation experiment

`profile_glm51_dsa_motivation.py` supersedes the older Figure 1 capacity and
dense-versus-sparse protocols below. It measures one TP=8 rank of a
GLM-5.1-shaped active top-2048 MLA working set with FlashInfer on the L40S, and
independently probes the exact allocation bytes of the 78-layer BF16 MLA plus
FP8 index-K state. The 30-GiB per-rank KV budget is a declared nominal H200
deployment assumption; 25/35/40 GiB sensitivity rows are emitted as modeled
capacity, not measured H200 configurations.

```bash
.venv/bin/python scripts/keye_trace/profile_glm51_dsa_motivation.py \
  --batches 1,2,4,8,9,16,32,48,64,96,128 \
  --contexts 32768,65536,131072,200000 \
  --capacity-budgets-gib 25,30,35,40 \
  --physical-probe-budget-gib 30 \
  --warmup-steps 100 \
  --measured-steps 500 \
  --independent-repeats 5 \
  --device 0 \
  --run-id <run-id>
```

Each performance point runs in an isolated process. The CUDA-event window is a
CUDA-graph replay of the FlashInfer BF16 MLA paged-decode kernel; planning,
index scoring, top-k selection, full-history allocation, communication, and
the rest of the transformer are excluded. The useful-FLOP numerator is
`2*B*8*2048*(576+512)`. Capacity workers really allocate all in-budget bytes,
but classify points beyond the declared budget as `over_budget`, never OOM.
Outputs remain under ignored `data/motivation/<run-id>/`.

## Figure 2 sparse-MLA SSD critical-path experiment

`profile_sparse_mla_ssd_path.py` measures the 78-layer GLM-5.1-shaped sparse
MLA path at fixed `B=8`. For each layer it submits the active-KV miss volume as
4-KiB random `O_DIRECT` reads through raw Linux `io_uring` at QD 128, waits for
the final completion, and then times one real FlashInfer FA2 BF16 top-2048 MLA
kernel. The path metric is the sum of those measured SSD waits and CUDA-event
kernel times. It is not full-model or end-to-end decode latency.

The host lacks liburing, so `io_uring_direct_reader.c` uses the stable kernel
ABI directly and is compiled into each run directory with `-Werror`. Prepare
the fully allocated 8-GiB NVMe backing file and run a short diagnostic with:

```bash
.venv/bin/python scripts/keye_trace/profile_sparse_mla_ssd_path.py \
  --diagnostic \
  --hit-rates 0,90,95,97,99,99.5,100 \
  --path-samples 3 \
  --independent-repeats 1 \
  --warmup-paths 5 \
  --io-warmup-layers 3 \
  --prepare-ssd-file \
  --run-id <diagnostic-run-id>
```

The formal protocol requires five isolated repeats and 100 path samples per
hit rate:

```bash
.venv/bin/python scripts/keye_trace/profile_sparse_mla_ssd_path.py \
  --hit-rates 0,90,95,97,99,99.5,100 \
  --path-samples 100 \
  --independent-repeats 5 \
  --warmup-paths 20 \
  --io-warmup-layers 10 \
  --run-id <formal-run-id>
```

Raw per-path samples, the pooled mean/P99 summary, job-level CSVs, compiled I/O
helper, and manifest are written under ignored `data/motivation/<run-id>/`.
Mean and P99 are normalized independently to their corresponding 100%-hit
compute-only values. The experiment excludes index scoring, top-k selection,
SSD-to-GPU transfer, communication, MoE, and other transformer work.

### Deprecated Figure 1 GPU profiles

The following protocols are retained only for traceability. They supported the
superseded GLM-5.2 dense-versus-DSA Figure 1 and must not be mixed into the new
GLM-5.1 capacity-concurrency claim.

#### GPU attention saturation profile for G2.5 Figure 1(b)

`profile_gpu_attention_saturation.py` compares the validated Keye exact-token
DSA Triton kernel with SGLang's dense Triton decode kernel using the same BF16
Keye GQA tensors. This is explicitly a GPU kernel proxy, not a GLM-5.2
FlashMLA measurement. Each point runs in an isolated process, records CUDA-event
latency plus a PyTorch Profiler trace, and preserves OOM points as
`capacity_infeasible`.

Smoke test one point before the full sweep:

```bash
.venv/bin/python scripts/keye_trace/profile_gpu_attention_saturation.py \
  --contexts 65536 \
  --batches 1 \
  --run-id <smoke-run-id>
```

Full Figure 1(b) sweep:

```bash
.venv/bin/python scripts/keye_trace/profile_gpu_attention_saturation.py \
  --contexts 65536,262144,1048576 \
  --batches 1,2,4,8,16,32,64,128 \
  --warmups 16 \
  --repeats 64 \
  --profiler-repeats 5 \
  --run-id <run-id>
```

The output CSV, per-point JSON, manifest, and profiler traces are written below
`data/gpu-attention-profile/<run-id>/` and remain outside Git.

Large flattened KV buffers require 64-bit K/V pointer offsets in both dense and
sparse Triton kernels. With 32-bit address multiplication, configurations at
or above 2^31 tensor elements can fail with an illegal memory access before HBM
is exhausted. The Figure 1 sweep also uses chunked tensor initialization to
avoid the corresponding large-fill limit in PyTorch 2.9.1+cu130.

On this host, the Kineto `torch.profiler` interface returns zero CUDA device
time, while the autograd profiler provides event-correlated device time. The
script therefore uses CUDA-event median/p10/p90 as the plotted metric and the
autograd profiler trace as an independent audit artifact.

#### GLM-5.2-shape utilization profile

`profile_glm52_dsa_utilization.py` is the current Figure 1(b) protocol. It
aligns both paths to GLM-5.2 absorbed MLA (Hq=64, Hkv=1, Q/K=576, V=512,
BF16): Dense reads the full token index, while the DSA proxy reads a
top-2048 token index through the same SGLang Triton decode kernel. The shared
kernel and KV buffer isolate the full-history versus sparse-gather difference.

```bash
PYTHONPATH=python .venv/bin/python \
  scripts/keye_trace/profile_glm52_dsa_utilization.py \
  --contexts 65536,262144,524288,1048576 \
  --batches 1,8,32,64 \
  --warmups 8 \
  --independent-repeats 5 \
  --window-seconds 1.5 \
  --sample-interval-ms 100 \
  --run-id <run-id>
```

The metric is NVIDIA NVML `utilization.gpu`, sampled by `nvidia-smi` during
five independent continuous-launch windows. It is a GPU compute duty-cycle
proxy, not occupancy, theoretical/effective TFLOPS, per-SM active cycles, or
target-NPU utilization. Both paths retain OOM points instead of filling or
extrapolating them.

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

Before collecting a large trace, validate the SM80 attention kernel directly
against a PyTorch reference on both target GPUs:

```bash
.venv/bin/python scripts/keye_trace/validate_keye_dsa_attention.py \
  --output data/kernel-validation/<run-id>/results.json
```

The validator covers BF16/FP16, decode split-K, prefill, ragged padded top-k,
and the physical KV-slot adapter at Keye's real 32-query-head, 4-KV-head,
128-head-dimension, K=2048 shape.

Every segmented or schema-intervention replay automatically audits the saved
outputs after inference. Existing runs can be checked independently with:

```bash
.venv/bin/python scripts/keye_trace/audit_inference_outputs.py \
  --run-dir data/agent-score-trace/<run-id>
```

The audit checks response/token metadata, finite latency, invalid characters,
pathological repetition, finish reason, and whether the expected BFCL tool and
ground-truth constants appear. Semantic checks are diagnostics rather than a
replacement for the official BFCL evaluator.

For a controlled trace-on versus trace-off smoke test, keep the model,
requests, random seed, and sampling settings fixed, then compare outputs with:

```bash
.venv/bin/python scripts/keye_trace/compare_inference_outputs.py \
  --reference-run data/<reference-run> \
  --candidate-run data/<trace-run> \
  --output data/<comparison.json>
```

The report includes exact equality and the common generated-token prefix. A
TP/MoE baseline can be non-deterministic even with greedy decoding, so exact
inequality must be interpreted against a repeated trace-off baseline.

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

For the schema-v4 semantic experiment, use the full-layer analyzer:

```bash
.venv/bin/python scripts/keye_trace/analyze_segment_trace.py \
  --run-dir data/agent-score-trace/<run-id> \
  --style /path/to/matplotlib_style.mplstyle
```

It emits per-layer step metrics, a sampled full 48-layer similarity matrix,
semantic segment lift and cross-round stable-core metrics, stripe diagnostics,
trace-driven candidate-set simulations, typed Parquet tables, and PDF/300 dpi
PNG figures. Statistical intervals are clustered by trajectory.

To sweep the previous-step score-ranked candidate size against the next-step
normal top-k, reuse the schema-v4 trace without starting a model server:

```bash
.venv/bin/python scripts/keye_trace/analyze_adjacent_step_k_sweep.py \
  --run-dir data/agent-score-trace/<run-id> \
  --style /path/to/matplotlib_style.mplstyle
```

The analysis excludes the newly appended decode position from historical-KV
coverage, audits score-derived K=2048 against the stored top-k, and reports
fixed-K/normalized-K curves, ECDFs, required-K distributions, per-layer
budgets, context-length sensitivity, and recency/random/oracle baselines.

To test whether next-step top-k additions follow an `i -> i + delta`
positional pattern, run the chance-corrected shift analysis:

```bash
.venv/bin/python scripts/keye_trace/analyze_topk_shift.py \
  --run-dir data/agent-score-trace/<run-id>
```

It conditions on positions outside the previous top-k, reports new-entry
precision lift against the same transition's background prevalence, and
compares shift-neighborhood unions with previous-score candidates at exactly
the same width. It writes trajectory-layer Parquet tables and PDF/300 dpi PNG
figures. To restyle existing figures without rereading trace chunks, append
`--plot-only`.

For the long-context compact-v5 gate, additionally test whether that token
signal survives fixed-page I/O and compare previous-score rank at equal token
and page budgets:

```bash
.venv/bin/python scripts/keye_trace/analyze_topk_shift_page_value.py \
  --run-dirs \
    data/compact-trace-v5/<ruler-32k-run> \
    data/compact-trace-v5/<ruler-64k-run> \
    data/compact-trace-v5/<ruler-128k-run> \
  --output-dir data/compact-trace-v5/<shift-page-analysis>
```

The token comparison uses all sampled layers and the exact compact top-4096.
The equal-page-budget comparison uses only layers with a retained full-score
shadow, because a top-4096 prefix may not contain enough distinct pages to
match the shift predictor's I/O budget. Newly appended decode positions are
excluded from both targets. Run `--self-test` for the synthetic metric check.

To measure whether semantic prompt regions receive disproportionate DSA
selection after controlling for region length and position, run:

```bash
.venv/bin/python scripts/keye_trace/analyze_prompt_segment_selection.py \
  --run-dir data/agent-score-trace/<run-id> \
  --style /path/to/matplotlib_style.mplstyle
```

This offline Phase-A analysis labels tool relevance and cross-round history,
reports natural-length and fixed-window metrics, constructs position-matched
controls, and simulates semantic candidate-KV placement. It evaluates the
pre-registered gate for a later controlled Phase-B trace collection. The
simulation marks policies that use current or future ground truth as oracles;
deployable policies use only type/age or previous-round selection frequency.

If Phase A triggers the controlled tool-schema validation, prepare the 72
strictly nested budget/position variants with:

```bash
.venv/bin/python scripts/keye_trace/run_bfcl_schema_intervention.py \
  --bfcl-root data/external/gorilla/berkeley-function-call-leaderboard \
  --source-run data/agent-score-trace/<semantic-run-id> \
  --output-dir data/agent-score-trace/<intervention-run-id> \
  --prepare-only
```

The intervention uses 12 single-target requests (three from each BFCL
category), schema budgets near 2.5k/3.5k/full tokens, and target-schema
front/tail placement. Distractor sets are deterministic and strictly nested;
each replay is cache-flushed and generates up to 256 greedy decode tokens,
stopping at EOS by default.

After replay, analyze the 48-layer intervention with:

```bash
.venv/bin/python scripts/keye_trace/analyze_schema_intervention.py \
  --run-dir data/agent-score-trace/<intervention-run-id> \
  --style /path/to/matplotlib_style.mplstyle
```

Decode step 0 is the primary paired comparison. Later steps are marked
comparable only while all six variants of a source request share the same
generated prefix.

## Candidate rescoring quality baseline

Greedy TP/MoE requests are not reliably reproducible across separate requests,
including duplicate rows submitted in one HTTP batch. Do not use
`run_teacher_forced_rescore.py` as an absolute decode-quality gate: its
single-token requests are evaluated from prefill and do not exercise the
decode-only rescoring path.

For a valid local quality check, set `KEYE_RESCORE_ATTN_SHADOW_DIR` and run a
continuous multi-token request. The server computes candidate and full-exact
attention from the same hidden state, aggregates the comparison over TP ranks,
and writes `attention_shadow.jsonl`. Keep
`KEYE_RESCORE_ATTN_SHADOW_EXACT_MAIN=1` to measure isolated per-layer error
without accumulating candidate perturbations; set it to `0` to run the actual
candidate path while retaining the exact shadow.

```bash
.venv/bin/python scripts/keye_trace/analyze_attention_shadow.py \
  --run-dir data/cross-layer-candidate-rescore/<run-id>/attention-events \
  --output-dir data/cross-layer-candidate-rescore/<run-id>/attention-analysis
```

`run_paired_decode_rescore.py` and `analyze_paired_decode_rescore.py` remain as
diagnostics for quantifying cross-request instability. They are not an
algorithm-quality baseline unless the exact/exact control gate passes.

For a longer output audit, `run_exact_reference_logprobs.py` also accepts
`--max-new-tokens` and `--respect-eos`. In that mode the old 32-token reference
is only a diagnostic prefix; task correctness must be judged from the saved
response and BFCL audit.

To screen a smaller K without rerunning the model, take the sorted prefix of an
existing larger-candidate trace. This measures exact top-2048 containment but
does not replace the same-hidden-state attention check:

```bash
.venv/bin/python scripts/keye_trace/analyze_candidate_prefix.py \
  --events-dir data/cross-layer-candidate-rescore/<run-id>/fidelity-events \
  --output-dir data/cross-layer-candidate-rescore/<run-id>/prefix-k2560 \
  --candidate-k 2560
```

For a static two-tier candidate budget, set the default and only list the
larger-K exceptions. For example, this uses K=2560 except at target layers 28
and 38:

```bash
export KEYE_RESCORE_CANDIDATE_K=2560
export KEYE_RESCORE_CANDIDATE_K_BY_LAYER=28:3072,38:3072
```

`prepare_cross_layer_lookahead.py --eligible-rank 1` materializes the
second-longest tool-bearing round from each trajectory. It is useful as a
correlated validation set after the longest rounds have been used for layer
selection; it is not a substitute for new trajectories.

### Analyze fixed-page expansion of token-level top-k

Before implementing page-based DRAM/SSD prefetch, measure how much a token-level
top-k expands when the storage system must fetch whole pages. The default is a
stratified seven-layer sample and the first round from every trajectory:

```bash
.venv/bin/python scripts/keye_trace/analyze_topk_page_locality.py \
  --run-dir data/agent-score-trace/2026-07-31_1106_bfcl-segmented-48l-v01
```

The analysis reports active-page fraction and read amplification together with
adjacent-step page reuse. High reuse is not treated as useful locality when each
step already touches nearly all pages. Use `--all-rounds` for the full run, or
override `--layers` and `--page-sizes` with comma-separated integer lists.

To test whether page co-activation adds predictive value above previous-score
K3072 at the same page capacity, run:

```bash
.venv/bin/python scripts/keye_trace/analyze_page_coactivation_increment.py \
  --run-dirs \
    data/agent-score-trace/<bfcl-run> \
    data/compact-trace-v5/<ruler-32k-run> \
    data/compact-trace-v5/<ruler-64k-run> \
    data/compact-trace-v5/<ruler-128k-run> \
  --output-dir data/agent-score-trace/<coactivation-analysis>
```

The online history-neighbor predictor uses only completed transitions from the
same trajectory. BFCL trajectories are split by category into calibration and
test halves; beta is selected on calibration only. Schema-v4 target top-k is
reconstructed exactly from retained full scores, while schema-v5 requires the
`torch_exact` backend. Use `--plot-only` to regenerate summaries/figures from
existing tables.

To convert exact compact-v5 traces into a capacity and transfer sensitivity
study for an HBM--DRAM--SSD hierarchy, run:

```bash
.venv/bin/python scripts/keye_trace/simulate_multitier_prefetch.py \
  --run-dirs \
    data/compact-trace-v5/<ruler-32k-run> \
    data/compact-trace-v5/<ruler-64k-run> \
    data/compact-trace-v5/<ruler-128k-run> \
  --output-dir data/compact-trace-v5/<multitier-simulation>
```

The simulator uses the previous step's exact score rank as the deployable
predictor, sweeps candidate K, page size, and tier capacity, and corrects every
miss on demand. It reports observed page/byte traffic separately from a
parameterized PCIe/NVMe time estimate. The estimate is not a measured speedup;
its purpose is to identify configurations worth implementing. Run
`--self-test` for cache-state and page-accounting checks, or `--plot-only` to
regenerate PDF and 300 dpi PNG figures from saved tables.

The loader also accepts a run containing multiple prepared requests. For a
cross-task comparison, use the dedicated analysis so task differences are not
hidden by a pooled average:

```bash
.venv/bin/python scripts/keye_trace/analyze_multitask_prefetch.py \
  --run-dirs data/compact-trace-v5/<ruler-multitask-run> \
  --output-dir data/compact-trace-v5/<multitask-analysis> \
  --page-size 4 \
  --hbm-logical-gib 1.2 \
  --dram-logical-gib 3.0
```

It writes per-transition, per-request-step, per-task, and overall tables; a
deadline-specific best-K table; and PDF/300 dpi PNG cross-task figures. Use
fixed logical GiB capacities when comparing different context lengths. If the
two absolute-capacity arguments are omitted, HBM and DRAM default to fractions
of each request's realized KV size and are only suitable for within-length
sensitivity analysis. The transfer-time columns remain model estimates rather
than speed results.

To separate candidate information from HBM admission cost, first compare
top-4096/6144/8192 under the same logical capacities:

```bash
.venv/bin/python scripts/keye_trace/analyze_large_candidate_gate.py \
  --run-dirs data/compact-trace-v5/<top8192-runs> \
  --output-dir data/compact-trace-v5/<large-candidate-analysis> \
  --candidate-k 4096,6144,8192
```

This analysis reports exact next-step candidate coverage together with useful,
unused, and pollution bytes from deterministic cache replay. It does not
report speedup. If wider full admission is worse, run Gate C0 with the exact
page-transfer count of the K4096 baseline fixed independently at every layer
and transition:

```bash
PYTHONPATH=scripts/keye_trace .venv/bin/python \
  scripts/keye_trace/analyze_equal_budget_admission.py \
  --run-dirs data/compact-trace-v5/<top8192-runs> \
  --output-dir data/compact-trace-v5/<equal-budget-analysis> \
  --layers 0,7,15,23,31,39,47 \
  --page-size 4 --hbm-logical-gib 1.2 --dram-logical-gib 3.0
```

Gate C0 uses RULER only to choose among predeclared online page features and
then evaluates the frozen choice on LongBench-v2. Exact top-2048 demand and
correction are unchanged. PDF and 300 dpi PNG figures are emitted alongside
transition, request, and dataset/context summaries.

To verify that trace collection did not alter saved generation results, run a
matched no-trace replay and compare the complete output token sequences:

```bash
.venv/bin/python scripts/keye_trace/compare_inference_runs.py \
  --reference-run data/compact-trace-v5/<no-trace-run> \
  --candidate-run data/compact-trace-v5/<trace-run> \
  --output data/compact-trace-v5/<trace-run>/analysis/trace-noninvasive-audit-v01/summary.json
```

The comparison requires identical request IDs and prompt hashes, then checks
output token IDs, decoded text, answer recall, and both runs' structural audit.

For a pre-generated RULER calibration/test split, select the same zero-based
row from every task with `--sample-start`. For example, use
`--sample-start 1 --samples-per-task 1` for calibration and reserve row 2 for
the final test.
The source ordinal is retained in both the request ID and run metadata.

If a long RULER run is interrupted, `audit_ruler_saved_runs.py` rebuilds one
structural and answer audit from the completed requests in the original and
continuation directories. Prepared-but-unfinished requests are excluded.

After calibration, freeze a context/deadline-to-K mapping in a JSON file and
evaluate it without test-time tuning:

```bash
.venv/bin/python scripts/keye_trace/evaluate_frozen_prefetch_policy.py \
  --analysis-dirs \
    data/compact-trace-v5/<64k-analysis> \
    data/compact-trace-v5/<128k-analysis> \
  --policy scripts/keye_trace/configs/ruler_prefetch_policy_v1.json \
  --split-label ordinal2-blind-test \
  --output-dir data/compact-trace-v5/<frozen-policy-evaluation>
```

The test-only oracle in this output is a diagnostic upper bound. It is never
used to change the frozen action selected from context length and deadline.

For an external LongBench-v2 test, first create one deterministic request per
official domain in each context bucket. Selection uses only Keye prompt token
length, domain, and stable source ID; it never uses the answer, model output,
or DSA trace:

```bash
.venv/bin/python scripts/keye_trace/run_longbench_v2_compact_trace.py \
  --dataset /Tan/dataset/LongBench-v2/0000.parquet \
  --output-dir data/compact-trace-v5/<longbench-v2-run> \
  --length-config 65536 \
  --prepare-only
```

After starting the compact-v5 server with request prefix `lbv2v5__`, replay
the frozen sidecar with `--reuse-prepared`. Add `--resume` to preserve and skip
request IDs already present in `requests.jsonl`. The runner retains the full
response and audits official multiple-choice extraction, exact prompt/output
token counts, finish reason, EOS placement, and response structure.

The Keye reasoning checkpoint emits a long `<think>` block before the answer,
so the formal trace protocol applies the model chat template and pre-fills the
answer-format prefix without supplying A--D. It also requires at least 32 new
tokens so every request yields one complete compact chunk per sampled layer:

```bash
.venv/bin/python scripts/keye_trace/run_longbench_v2_compact_trace.py \
  --dataset /Tan/dataset/LongBench-v2/0000.parquet \
  --output-dir data/compact-trace-v5/<longbench-v2-run> \
  --length-config 65536 \
  --max-new-tokens 128 \
  --min-new-tokens 32 \
  --apply-chat-template \
  --answer-first-system-prompt \
  --answer-assistant-prefix \
  --reuse-prepared \
  --resume
```

The fixed assistant prefix is `</think>\nThe correct answer is (`. It fixes
only response format; the model still generates the answer letter. Accuracy is
reported independently from trace validity and is never used to tune the
frozen prefetch policy.

## Real prefetch-deadline tracing

Use CUDA events to measure how long a deployable predictor has before its KV
candidate must be consumed. The trace has two distinct timing contracts:

- `previous_step_same_layer`: previous decode step's top-k-ready point at layer
  `l` to the next step's attention consumption at the same layer. This is the
  storage-prefetch window for a previous-step score predictor.
- `cross_layer_candidate`: a lookahead candidate becoming ready to the target
  layer's attention consumption in the same step. This is a cross-layer
  computation window and must not be pooled with the previous-step window.

Start an eager-mode server with tracing enabled:

```bash
export KEYE_DEADLINE_TRACE_DIR=/absolute/path/to/run/events
export KEYE_DEADLINE_TRACE_LAYERS=0,7,15,23,31,39,47
export KEYE_DEADLINE_TRACE_RID_PREFIX=deadline-lbv2
export KEYE_DEADLINE_TRACE_INTERVALS=32
export KEYE_DEADLINE_TRACE_PENDING_LIMIT=8192
```

Deadline tracing intentionally rejects CUDA Graph capture. CUDA events are
recorded on the active stream without a device-wide synchronization in the hot
path; completed intervals are written by a background JSONL thread. Generate
at least one more token than the requested interval count so the final event
has a later consumption point.

On this host, make CUDA enumeration follow PCI bus order before selecting the
L40S and A6000. Without `CUDA_DEVICE_ORDER=PCI_BUS_ID`, ordinal 1 can resolve to
the RTX 3090 and create an invalid TP memory pairing:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,2
```

Replay retained requests, including their complete inference responses:

```bash
.venv/bin/python scripts/keye_trace/run_deadline_trace_requests.py \
  --prepared-requests data/compact-trace-v5/<run>/prepared_requests.jsonl \
  --output-dir data/deadline-trace/<run>/requests \
  --rid-prefix deadline-lbv2-64k \
  --request-limit 6 \
  --min-new-tokens 36 \
  --max-new-tokens 36 \
  --flush-cache
```

Pair TP ranks conservatively using the minimum usable window, validate exact
coverage, and emit PDF plus 300 dpi PNG figures:

```bash
.venv/bin/python scripts/keye_trace/analyze_deadline_trace.py \
  --trace-dir data/deadline-trace/<run>/events \
  --output-dir data/deadline-trace/<run>/analysis-64k \
  --rid-prefix deadline-lbv2-64k \
  --context-label 64K \
  --expected-layers 0,7,15,23,31,39,47 \
  --expected-kinds previous_step_same_layer \
  --expected-intervals 32
```

For a matched trace-off/trace-on perturbation gate, collect streaming requests
with the same model, prompts and generation length, then run:

```bash
.venv/bin/python scripts/keye_trace/analyze_deadline_overhead.py \
  --baseline data/deadline-trace/<baseline>/requests.jsonl \
  --instrumented data/deadline-trace/<instrumented>/requests.jsonl \
  --output-dir data/deadline-trace/<audit>
```

This check reports client-streamed token intervals and exact output-token
agreement. It is only an instrumentation perturbation gate, not a speedup
measurement. Run the synthetic parser/plot integration test with:

```bash
.venv/bin/python scripts/keye_trace/test_deadline_trace_analysis.py
```

After collecting real LongBench-v2 windows, join them with the already frozen
fixed-capacity transfer tables instead of rerunning or retuning the policy:

```bash
.venv/bin/python scripts/keye_trace/analyze_measured_deadline_saturation.py \
  --request-step-tables \
    data/compact-trace-v5/<64k-analysis>/tables/by_request_step.parquet \
    data/compact-trace-v5/<128k-analysis>/tables/by_request_step.parquet \
  --deadline-intervals \
    data/deadline-trace/<deadline-summary>/combined_paired_intervals.csv \
  --output-dir data/deadline-trace/<saturation-analysis>
```

The join is exact at `(context, task, decode transition)` granularity. It takes
the minimum measured window over both TP ranks and all seven sampled layers,
then compares that conservative window with the aggregate modeled transfer for
the same transition. Candidate actions stay fixed at K=2048/2560/3072/4096;
20/50/100 ms are sensitivity points, and `measured` uses the actual CUDA-event
window. The output remains a transfer-model analysis, not measured speedup.
Run `--self-test` to verify alignment, stall arithmetic, and figure generation.

### Concurrency timing gate

`run_deadline_trace_requests.py` can issue a real concurrent wave while keeping
one retained response per request. `--request-indices` selects an explicit
prepared-request subset; `--concurrency` controls the wave width. Cache flush
occurs once before each wave, not between requests in the same wave:

```bash
.venv/bin/python scripts/keye_trace/run_deadline_trace_requests.py \
  --prepared-requests data/compact-trace-v5/<run>/prepared_requests.jsonl \
  --output-dir data/deadline-trace/<run>/requests-c4 \
  --rid-prefix deadline-concurrency \
  --request-indices 0,1,2,3 \
  --concurrency 4 \
  --min-new-tokens 36 \
  --max-new-tokens 36 \
  --flush-cache
```

The analyzer must be told the requested concurrency. It rejects a trace when
the observed decode `batch_size` does not sustain that value:

```bash
.venv/bin/python scripts/keye_trace/analyze_deadline_trace.py \
  --trace-dir data/deadline-trace/<run>/events \
  --output-dir data/deadline-trace/<run>/analysis-c4 \
  --rid-prefix deadline-concurrency \
  --context-label 32K-c4 \
  --requested-concurrency 4 \
  --expected-layers 0,7,15,23,31,39,47 \
  --expected-kinds previous_step_same_layer \
  --expected-intervals 32
```

Use `analyze_concurrency_deadline_gate.py` to compare only the common tasks
across runs, audit retained answer-first outputs, and record an unmeasured
configuration as capacity-blocked when its prompt plus output reservation
exceeds the server KV token pool. Its figure distinguishes CUDA-event timing
from the capacity model and does not report latency, throughput, or speedup.
The local concurrent-wave integration test is:

```bash
.venv/bin/python scripts/keye_trace/test_deadline_request_concurrency.py
```

## Gate E0 cross-dataset generalization

Gate E0 combines a 24-request all-48-layer paired set with a frozen expanded
set from RULER, LongBench-v2, and InfiniteBench. Expanded traces retain layers
`0,7,15,23,31,39,47`; nearest-layer weights are calibrated against the paired
48-layer measurements. Never label the seven-layer numbers as measurements.

After collection, first audit request/output/manifest coverage and then run the
compact trace validator. The frozen expanded contract is 57 requests, seven
chunks per request, exact K=4096, and 32 contiguous decode steps:

```bash
.venv/bin/python scripts/keye_trace/audit_gate_e0_collection.py \
  --run-dir data/compact-trace-v5/<gate-e0-run>

.venv/bin/python scripts/keye_trace/validate_compact_trace_v5.py \
  --run-dir data/compact-trace-v5/<gate-e0-run> \
  --expected-compact-k 4096 \
  --expected-threshold-ranks 2048,2560,3072,4096 \
  --expected-layers 0,7,15,23,31,39,47 \
  --expected-steps 32 \
  --expected-requests 57
```

Audit retained outputs independently from trace validity, then compute
request- and task-cluster bootstrap intervals and publication figures:

```bash
.venv/bin/python scripts/keye_trace/analyze_layer_sampling_bias.py \
  --run-dir data/compact-trace-v5/<gate-e0-run>/full48-paired \
  --output-dir data/compact-trace-v5/<gate-e0-run>/full48-paired/analysis/layer-sampling-bias-p4-v01 \
  --page-size 4 \
  --sampled-layers 0,7,15,23,31,39,47

.venv/bin/python scripts/keye_trace/audit_gate_e0_outputs.py \
  --run-dir data/compact-trace-v5/<gate-e0-run> \
  --output-dir data/compact-trace-v5/<gate-e0-run>/analysis/output-quality-v01

.venv/bin/python scripts/keye_trace/analyze_gate_e0_generalization.py \
  --run-dir data/compact-trace-v5/<gate-e0-run> \
  --page-size 4 \
  --full48-table data/compact-trace-v5/<gate-e0-run>/full48-paired/analysis/layer-sampling-bias-p4-v01/tables/by_request_layer_step.parquet \
  --output-dir data/compact-trace-v5/<gate-e0-run>/analysis/generalization-v01

.venv/bin/python scripts/keye_trace/audit_gate_e0_analysis.py \
  --analysis-dir data/compact-trace-v5/<gate-e0-run>/analysis/generalization-v01 \
  --full48-table data/compact-trace-v5/<gate-e0-run>/full48-paired/analysis/layer-sampling-bias-p4-v01/tables/by_request_layer_step.parquet \
  --full48-summary data/compact-trace-v5/<gate-e0-run>/full48-paired/analysis/layer-sampling-bias-p4-v01/summary.json
```

`plot_gate_e0_generalization.py` exports PDF and 300 dpi PNG. Its marker shape
distinguishes the all-48-layer measurement from the seven-layer estimate.
The formal storage-facing page metric uses the already selected 4-token page;
the older 64-token layer-sampling analysis is not used for Gate E0 claims.

## Gate E1 recurrent-resident protection

Gate E1 keeps the K4096 candidate pool fixed. The frozen v1 rule protects HBM
pages that appeared in at least two of the previous four completed exact
top-2048 page sets. It may replace low-priority K4096 pages only with pages
already resident in HBM, and it enforces page and byte budgets against the
unprotected K4096 counterfactual in the same cache state.

```bash
.venv/bin/python scripts/keye_trace/analyze_resident_page_protection.py \
  --run-dir data/compact-trace-v5/<gate-e0-run> \
  --policy scripts/keye_trace/configs/gate_e1_resident_protection_v1.json \
  --output-dir data/compact-trace-v5/<gate-e0-run>/analysis/gate-e1-v01

.venv/bin/python scripts/keye_trace/audit_resident_page_protection.py \
  --analysis-dir data/compact-trace-v5/<gate-e0-run>/analysis/gate-e1-v01 \
  --policy scripts/keye_trace/configs/gate_e1_resident_protection_v1.json

.venv/bin/python scripts/keye_trace/plot_resident_page_protection.py \
  --analysis-dir data/compact-trace-v5/<gate-e0-run>/analysis/gate-e1-v01 \
  --output-dir data/compact-trace-v5/<gate-e0-run>/analysis/gate-e1-v01/figures
```

The policy JSON was frozen before Gate E1 replay. A failed gate must be
reported as failed; do not retune it on the expanded test traces. Outputs are
shadow cache-quality and transfer-volume results, not measured speedup.

Because cache-policy deltas are nonlinear, calibrate the seven-layer Gate E1
estimate on the paired all-layer run. Replay the same 24 requests once with
the sampled layers and once with `--layers 0,1,...,47`, then compare them:

```bash
.venv/bin/python scripts/keye_trace/analyze_resident_protection_layer_sampling.py \
  --full48-analysis data/compact-trace-v5/<full48-e1-analysis> \
  --sampled-analysis data/compact-trace-v5/<seven-layer-e1-analysis> \
  --output-dir data/compact-trace-v5/<e1-layer-calibration>
```

`plot_resident_protection_layer_sampling.py` shows both group-level sampling
bias and per-request seven-layer versus full48 agreement. The paper must use
the full48 replay for paired core claims and attach the measured calibration
error to expanded seven-layer estimates.

Use `analyze_resident_protection_layer_profile.py` and
`plot_resident_protection_layer_profile.py` only on the paired full48 replay
to diagnose which layers move in a harmful direction. This is a post-hoc
explanation table, not permission to retune the frozen external-test policy.

`export_gate_e0_e1_core_data.py` refuses to export research-note CSVs unless
Gate E0 is complete at p4, retained-output quality passes, both Gate E1 runs
pass the independent output audit, all 48 layers are present in the paired
run, and all request pairs are matched in the layer-sampling calibration.
