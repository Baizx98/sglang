"""Benchmark the same BF16 scorer with raw, fused, and materialized key loads."""

from optimize import _selector, prepare_ablation


def prepare(q, key_cache, page_table, weights, lengths, scale):
    return prepare_ablation(
        q, key_cache, page_table, weights, lengths, scale, _selector(lengths)
    )
