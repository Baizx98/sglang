#!/usr/bin/env bash
set -uo pipefail

invocation_directory=$PWD
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo" || exit 1
scripts="$repo/scripts/indexer-optimization"
data="$repo/data/indexer-optimization"
python_executable="$repo/.venv/bin/python"
output=""
core_only=false

usage() {
    printf '%s\n' "Usage: $0 [--output-dir NEW_DIRECTORY] [--core-only]"
    printf '%s\n' "Set INDEXER_GPU, INDEXER_CUDA_HOME, INDEXER_FI074 and INDEXER_FI_DSL to override local defaults."
}

while (($#)); do
    case "$1" in
        --output-dir)
            if (($# < 2)); then usage >&2; exit 2; fi
            output=$2
            shift 2
            ;;
        --core-only)
            core_only=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z $output ]]; then
    output="$data/results/suite-$(date +%Y%m%d-%H%M%S)"
elif [[ $output != /* ]]; then
    output="$invocation_directory/$output"
fi
if ! mkdir -p "$(dirname "$output")" || ! mkdir "$output"; then
    printf 'Choose a new output directory. Existing results were preserved.\n' >&2
    exit 2
fi
output=$(cd "$output" && pwd)

export CUDA_VISIBLE_DEVICES="${INDEXER_GPU:-GPU-19717dbe-cbf8-b0b8-744d-1cb9a9365822}"
export CUDA_HOME="${INDEXER_CUDA_HOME:-$data/cuda130/nvidia/cu13}"
export INDEXER_FI074="${INDEXER_FI074:-$data/fi074}"
export INDEXER_FI_DSL="${INDEXER_FI_DSL:-$data/fi-dsl}"
export INDEXER_BENCH_SCRIPTS="$scripts"
export PATH="$CUDA_HOME/bin:$PATH"
export LIBRARY_PATH="$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CPATH="$CUDA_HOME/include/cccl:$repo/.venv/lib/python3.12/site-packages/nvidia/cu13/include${CPATH:+:$CPATH}"
export FLASHINFER_CUDA_ARCH_LIST=8.6

common=(--module "$scripts/vllm_baseline.py" --module "$scripts/optimize.py")
full=(--full --direct-packed --stages --packed-audit --align-width
      --select 'opt_(qk_n(64w4|128w4)|tail_runs)')
failures=0

run() {
    local release=$1
    local name=$2
    shift 2
    local -a command=("$python_executable")
    if [[ $release == 074 ]]; then
        command+=(-c 'import os, pathlib, runpy, sys
dsl = pathlib.Path(os.environ["INDEXER_FI_DSL"])
sys.path[:0] = [os.environ["INDEXER_FI074"], str(dsl / "nvidia_cutlass_dsl/dsl_packages"), str(dsl), os.environ["INDEXER_BENCH_SCRIPTS"]]
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")')
    fi
    printf 'Running %s\n' "$name"
    printf '%q ' "${command[@]}" "$scripts/bench.py" "$@" --output "$output/$name.jsonl" >> "$output/commands.sh"
    printf '\n' >> "$output/commands.sh"
    "${command[@]}" "$scripts/bench.py" "$@" --output "$output/$name.jsonl" > "$output/$name.log" 2>&1
    local result=$?
    printf '%s\t%s\t%d\n' "$(date -Is)" "$name" "$result" >> "$output/status.tsv"
    if ((result)); then
        ((failures+=1))
        printf 'See %s/%s.log\n' "$output" "$name" >&2
    fi
}

# Run only one GPU process at a time. Each complete-Indexer release uses the same shapes.
run 074 heldout-new-append --source real --batch 4 16 --length 131072 --layer 24 47 \
    --step 0 --append-steps 1 2 3 4 --request-offset 8 "${common[@]}"
run 068 heldout-old-full128 --source real --batch 4 16 --length 131072 --layer 36 47 \
    --step 2 4 --request-offset 8 "${common[@]}" "${full[@]}"
run 074 heldout-new-full128 --source real --batch 4 16 --length 131072 --layer 36 47 \
    --step 2 4 --request-offset 8 --ampere-cute "${common[@]}" "${full[@]}"
run 068 heldout-old-full-short --source real --batch 4 16 --length 32768 65536 --layer 36 47 \
    --step 2 4 --request-offset 8 "${common[@]}" "${full[@]}"
run 074 heldout-new-full-short --source real --batch 4 16 --length 32768 65536 --layer 36 47 \
    --step 2 4 --request-offset 8 --ampere-cute "${common[@]}" "${full[@]}"
run 068 heldout-old-mixed --source mixed --layer 47 --step 4 --request-offset 8 \
    --mixed-counts 4 4 8 --shuffle-block 0 128 1 "${common[@]}" "${full[@]}"
run 074 heldout-new-mixed --source mixed --layer 47 --step 4 --request-offset 8 \
    --mixed-counts 4 4 8 --shuffle-block 0 128 1 --ampere-cute "${common[@]}" "${full[@]}"
run 074 heldout-cold --source real --batch 1 --length 32768 --layer 36 47 --step 2 \
    --request-offset 8 --rotate 4 --ampere-cute --select 'opt_(qk_n(64w4|128w4)|tail_runs)' "${common[@]}"
run 074 heldout-warm --source real --batch 1 --length 32768 --layer 36 47 --step 2 \
    --request-offset 8 --ampere-cute --select 'opt_(qk_n(64w4|128w4)|tail_runs)' "${common[@]}"
run 074 holdout-replay-truth --source real --batch 4 16 --length 32768 65536 131072 \
    --layer 36 47 --step 2 4 --request-offset 8 --projection-mode captured-singletons \
    --check-only --select 'opt_(qk_n(64w4|128w4)|tail_runs|pack12_tail)' "${common[@]}"
run 074 kernel-launches --source real --batch 16 --length 131072 --layer 36 --step 2 \
    --request-offset 8 --ampere-cute --profile-kernels "${common[@]}" "${full[@]}"

if [[ $core_only == false ]]; then
    run 068 synthetic --source synthetic --batch 1 4 16 --length 32768 65536 131072 \
        --select 'opt_(qk_n(64w4|128w4|256w4|128w4cg)|runs|tail_runs|pack12_(n128|tail))' "${common[@]}"
    run 068 ties --source synthetic --batch 4 --length 32768 --variant ragged all_ties boundary_ties \
        --shuffle-block 0 1 --check-only --correctness-repeats 10 "${common[@]}"
    run 068 real-primary --source real --batch 1 4 16 --length 32768 65536 131072 --layer 12 --step 0 \
        --select 'opt_(qk|runs|tail_runs|static_runs)' "${common[@]}"
    run 068 rotation --source real --batch 1 --length 32768 --layer 12 --step 0 --rotate 4 "${common[@]}"
    run 074 int8-a6000 --source real --batch 1 16 --length 32768 131072 --layer 12 --step 0 \
        --ampere-cute --stages --module "$scripts/vllm_baseline.py" --module "$scripts/segments.py" \
        --select 'int8_|bf16_sample4'
    run 074 torch-b0 --source real --batch 1 --length 32768 --layer 12 --step 0 --torch-score-baseline \
        --ampere-cute "${common[@]}" --select 'a^'
    run 074 ablation-packed --source real --batch 4 16 --length 131072 --layer 36 --step 2 \
        --request-offset 8 --ampere-cute --module "$scripts/vllm_baseline.py" \
        --module "$scripts/packed_ablation.py" --select '^ablate_'
    run 074 fused-real --source real --batch 4 16 --length 131072 --layer 36 --step 2 \
        --request-offset 8 --ampere-cute --module "$scripts/vllm_baseline.py" \
        --module "$scripts/fused_threshold.py" --select '^fused_threshold_exact_flashinfer$'
    run 074 fused-ties --source synthetic --batch 4 --length 32768 --variant all_ties boundary_ties \
        --check-only --correctness-repeats 10 --ampere-cute --module "$scripts/vllm_baseline.py" \
        --module "$scripts/fused_threshold.py" --select '^fused_threshold_exact_flashinfer$'
fi

"$python_executable" "$scripts/summarize_holdout.py" "$output" || ((failures+=1))
printf 'Results %s\n' "$output"
if ((failures)); then
    exit 1
fi
