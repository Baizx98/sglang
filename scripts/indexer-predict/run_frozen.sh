#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$script_dir/../.." && pwd)
cd "$repo"

stage=${1:?choose calibration, test, test-native-int4 or test-native-warp}
run=${2:?provide an output directory}
data=${3:-data/indexer-predict}
books="$data/pq-codebooks-v1"
mkdir -p "$run"
python="$repo/.venv/bin/python"

bench() {
    INDEXER_FULL_TARGET="$script_dir/benchmark.py" "$repo/scripts/indexer-round2/run_full.sh" \
        --layers 1 12 24 47 --steps 2 4 --candidates 3072 8192 \
        --fused-query --direct-paged-build --fp16-scores \
        --quant-formats 4x1 4x2 5x1 5x2 6x1 8x1 --fit-int4 2 --fp4 --pair-pq "$books" \
        --progressive 1.25 1.5 2 --quant-tiles 128 256 "$@"
}

case "$stage" in
    calibration)
        bench --capture "$data/capture-64-128k-calibration" \
            --length 65536 131072 --batch 4 --output "$run/calibration-shorter"
        bench --capture "$data/capture-near256-gate-v2" "$data/capture-near256-remaining" \
            --length 262144 --batch 4 --output "$run/calibration-near256"
        "$python" "$script_dir/policy.py" freeze --output "$run/policy.json" --inputs \
            "$run/calibration-shorter/results.jsonl" "$run/calibration-near256/results.jsonl"
        ;;
    test|test-native-int4|test-native-warp)
        if [[ ! -s "$run/policy.json" ]]; then
            echo "Calibration policy is required before measuring the frozen holdouts." >&2
            exit 1
        fi
        extra=()
        near_extra=()
        case "$stage" in
            test-native-int4)
                extra=(--native-int4 --native-int4-fused-query)
                ;;
            test-native-warp)
                extra=(--native-int4 --native-int4-fused-query --native-int8)
                near_extra=(--native-int8)
                ;;
        esac
        bench --capture "$data/capture-64-128k-test" --split test "${extra[@]}" \
            --length 65536 131072 --batch 4 16 --output "$run/test-shorter"
        bench --capture "$data/capture-near256-remaining" "$data/capture-near256-test" --split test "${near_extra[@]}" \
            --length 262144 --batch 4 16 --output "$run/test-near256"
        "$python" "$script_dir/policy.py" test --policy "$run/policy.json" \
            --output "$run/holdout.json" --inputs \
            "$run/test-shorter/results.jsonl" "$run/test-near256/results.jsonl"
        ;;
    *)
        echo "Unknown benchmark stage: $stage" >&2
        exit 2
        ;;
esac
