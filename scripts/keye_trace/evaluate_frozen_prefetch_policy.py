#!/usr/bin/env python3
"""Evaluate a frozen context/deadline candidate-K policy without test-time tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

TIME_MODEL_CONTRACT = {
    "kv_bytes_per_token_layer": 2048,
    "pcie_gbps": 25.0,
    "ssd_gbps": 7.0,
    "pcie_latency_us": 10.0,
    "ssd_latency_us": 100.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(path.read_text())
    if policy.get("uses_task_label") or policy.get("uses_test_trace"):
        raise ValueError("frozen policy must not use task labels or test traces")
    return policy


def request_means(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    keys = [
        "request_id",
        "task",
        "source_index",
        "context_config",
        "method",
        "candidate_k",
    ]
    return frame.groupby(keys, as_index=False, dropna=False)[metric].mean()


def mapping_deadlines(policy: dict[str, Any]) -> set[float]:
    return {
        float(deadline)
        for deadlines in policy["mapping"].values()
        for deadline in deadlines
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--split-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    policy = load_policy(args.policy)
    frames: list[pd.DataFrame] = []
    owners: list[tuple[str, int]] = []
    inputs: list[dict[str, Any]] = []
    contract = policy["simulation_contract"]
    for index, analysis_dir in enumerate(args.analysis_dirs):
        path = analysis_dir / "tables" / "by_request_step.parquet"
        summary_path = analysis_dir / "summary.json"
        reproducibility_path = analysis_dir / "reproducibility.json"
        analysis_summary = json.loads(summary_path.read_text())
        reproducibility = json.loads(reproducibility_path.read_text())
        analysis_args = reproducibility["arguments"]
        setting = analysis_summary["setting"]
        observed_hash = sha256(path)
        declared_hash = analysis_summary["artifacts"].get(
            "tables/by_request_step.parquet"
        )
        contract_checks = {
            "artifact_hash_matches_analysis_summary": observed_hash == declared_hash,
            "fixed_capacity_mode": setting.get("capacity_mode")
            == "fixed logical bytes across the full model",
            "page_size_tokens": int(setting["page_size_tokens"])
            == int(contract["page_size_tokens"]),
            "hbm_logical_gib": math.isclose(
                float(setting["hbm_logical_gib"]),
                float(contract["hbm_logical_gib"]),
                rel_tol=0,
                abs_tol=1e-12,
            ),
            "dram_logical_gib": math.isclose(
                float(setting["dram_logical_gib"]),
                float(contract["dram_logical_gib"]),
                rel_tol=0,
                abs_tol=1e-12,
            ),
            "sampled_layers": list(setting["sampled_layers"])
            == list(contract["sampled_layers"]),
            "scaled_model_layers": int(setting["scaled_model_layers"])
            == int(contract["scaled_model_layers"]),
            "tensor_parallel_size": int(setting["tensor_parallel_size"])
            == int(contract["tensor_parallel_size"]),
            "overlap_windows_ms": set(float(value) for value in mapping_deadlines(policy))
            <= set(float(value) for value in setting["overlap_windows_ms"]),
            "candidate_k_actions": [int(value) for value in analysis_args["candidate_k"]]
            == [int(value) for value in policy["candidate_k_actions"]],
            "requested_hbm_logical_gib": math.isclose(
                float(analysis_args["hbm_logical_gib"]), 1.2, rel_tol=0, abs_tol=1e-12
            ),
            "time_model": all(
                math.isclose(
                    float(analysis_args[name]), float(expected), rel_tol=0, abs_tol=1e-12
                )
                for name, expected in TIME_MODEL_CONTRACT.items()
            ),
        }
        if not all(contract_checks.values()):
            failed = [name for name, passed in contract_checks.items() if not passed]
            raise ValueError(f"{analysis_dir}: simulation contract mismatch: {failed}")
        frame = pd.read_parquet(path)
        frames.append(frame)
        owners.extend((request_id, index) for request_id in frame.request_id.unique())
        inputs.append(
            {
                "path": str(path.resolve()),
                "sha256": observed_hash,
                "analysis_summary": str(summary_path.resolve()),
                "analysis_summary_sha256": sha256(summary_path),
                "reproducibility": str(reproducibility_path.resolve()),
                "reproducibility_sha256": sha256(reproducibility_path),
                "contract_checks": contract_checks,
            }
        )
    owner_frame = pd.DataFrame(owners, columns=["request_id", "owner"])
    if owner_frame.request_id.duplicated().any():
        duplicate = owner_frame[owner_frame.request_id.duplicated()].request_id.iloc[0]
        raise ValueError(f"request appears in multiple analysis inputs: {duplicate}")
    data = pd.concat(frames, ignore_index=True)

    rows: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    mapping = policy["mapping"]
    actions = [int(value) for value in policy["candidate_k_actions"]]
    for context_key, deadlines in mapping.items():
        context = int(context_key)
        context_data = data[data.context_config == context]
        if context_data.empty:
            raise ValueError(f"no analysis input for context {context}")
        for deadline_key, selected_k_value in deadlines.items():
            deadline = float(deadline_key)
            selected_k = int(selected_k_value)
            metric = f"stall_{deadline:g}ms"
            if metric not in context_data:
                raise ValueError(f"missing deadline metric {metric}")
            means = request_means(context_data, metric)
            lru = means[means.method == "lru-demand"][
                ["request_id", "task", "source_index", metric]
            ].rename(columns={metric: "lru_stall_ms"})
            candidates = means[means.method == "previous-score"]
            selected = candidates[candidates.candidate_k == selected_k][
                ["request_id", "task", "source_index", metric]
            ].rename(columns={metric: "policy_stall_ms"})
            if len(selected) != len(lru):
                raise ValueError(f"incomplete selected K={selected_k} for context {context}")
            joined = lru.merge(
                selected,
                on=["request_id", "task", "source_index"],
                validate="one_to_one",
            )
            candidate_wide = candidates.pivot(
                index=["request_id", "task", "source_index"],
                columns="candidate_k",
                values=metric,
            )
            missing_actions = sorted(
                set(actions) - set(int(value) for value in candidate_wide.columns)
            )
            if missing_actions:
                raise ValueError(f"missing candidate actions {missing_actions}")
            oracle = (
                candidate_wide[actions]
                .min(axis=1)
                .rename("oracle_stall_ms")
                .reset_index()
            )
            joined = joined.merge(
                oracle,
                on=["request_id", "task", "source_index"],
                validate="one_to_one",
            )
            joined["split_label"] = args.split_label
            joined["context_config"] = context
            joined["overlap_deadline_ms"] = deadline
            joined["policy_candidate_k"] = selected_k
            joined["reduction_vs_lru"] = (
                1 - joined.policy_stall_ms / joined.lru_stall_ms
            )
            joined["regret_vs_test_oracle_ms"] = (
                joined.policy_stall_ms - joined.oracle_stall_ms
            )
            request_rows.extend(joined.to_dict("records"))

            rows.append(
                {
                    "split_label": args.split_label,
                    "context_config": context,
                    "overlap_deadline_ms": deadline,
                    "policy_candidate_k": selected_k,
                    "requests": len(joined),
                    "lru_stall_ms_mean": float(joined.lru_stall_ms.mean()),
                    "policy_stall_ms_mean": float(joined.policy_stall_ms.mean()),
                    "policy_stall_ms_p95": float(joined.policy_stall_ms.quantile(0.95)),
                    "reduction_vs_lru_mean": float(
                        1 - joined.policy_stall_ms.mean() / joined.lru_stall_ms.mean()
                    ),
                    "no_worse_request_fraction": float(
                        (joined.policy_stall_ms <= joined.lru_stall_ms).mean()
                    ),
                    "worst_request_reduction_vs_lru": float(
                        joined.reduction_vs_lru.min()
                    ),
                    "test_oracle_stall_ms_mean": float(joined.oracle_stall_ms.mean()),
                    "policy_regret_vs_test_oracle_ms_mean": float(
                        joined.regret_vs_test_oracle_ms.mean()
                    ),
                }
            )
            for candidate_k in actions:
                values = candidate_wide[candidate_k]
                baselines.append(
                    {
                        "split_label": args.split_label,
                        "context_config": context,
                        "overlap_deadline_ms": deadline,
                        "candidate_k": candidate_k,
                        "stall_ms_mean": float(values.mean()),
                        "reduction_vs_lru_mean": float(
                            1 - values.mean() / joined.lru_stall_ms.mean()
                        ),
                    }
                )

    overall = pd.DataFrame(rows).sort_values(
        ["context_config", "overlap_deadline_ms"]
    )
    by_request = pd.DataFrame(request_rows).sort_values(
        ["context_config", "overlap_deadline_ms", "task", "request_id"]
    )
    fixed = pd.DataFrame(baselines).sort_values(
        ["context_config", "overlap_deadline_ms", "candidate_k"]
    )
    expected_contexts = sorted(int(value) for value in mapping)
    observed_contexts = sorted(int(value) for value in data.context_config.unique())
    gates = {
        "all_input_contracts_match": all(
            all(item["contract_checks"].values()) for item in inputs
        ),
        "all_policy_contexts_evaluated": observed_contexts == expected_contexts,
        "all_selected_actions_available": len(overall)
        == sum(len(value) for value in mapping.values()),
        "all_requests_no_worse_than_lru": bool(
            (by_request.policy_stall_ms <= by_request.lru_stall_ms).all()
        ),
        "positive_mean_reduction_for_deadlines_above_1ms": bool(
            (
                overall[overall.overlap_deadline_ms > 1].reduction_vs_lru_mean
                > 0
            ).all()
        ),
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    overall_path = output / "summary_overall.csv"
    request_path = output / "by_request.csv"
    fixed_path = output / "fixed_k_baselines.csv"
    overall.to_csv(overall_path, index=False)
    by_request.to_csv(request_path, index=False)
    fixed.to_csv(fixed_path, index=False)
    summary = {
        "schema_version": 1,
        "analysis_kind": "frozen-policy evaluation; test oracle is diagnostic only",
        "split_label": args.split_label,
        "policy": str(args.policy.resolve()),
        "policy_sha256": sha256(args.policy),
        "inputs": inputs,
        "gates": gates,
        "passed": all(gates.values()),
        "overall": overall.to_dict("records"),
        "artifacts": {
            overall_path.name: sha256(overall_path),
            request_path.name: sha256(request_path),
            fixed_path.name: sha256(fixed_path),
        },
        "limitations": [
            "Transfer time is modeled rather than measured.",
            "The test oracle uses future test results and is not deployable.",
            "A successful gate validates robustness of the frozen mapping, not real speedup.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
