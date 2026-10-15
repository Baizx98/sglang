"""Independent exact-state DP and adversarial version tests; CPU only."""

import itertools
import random
import tempfile
from functools import lru_cache

import numpy as np
from horizonkv_belady import compile_belady, pack, run


def exact(demands, writes, capacity):
    @lru_cache(None)
    def dp(t, resident):
        if t == len(demands):
            return 0
        resident = set(resident) - writes[t]
        hits = len(resident & demands[t])
        pool = sorted(resident | demands[t])
        return hits + max(
            dp(t + 1, subset)
            for n in range(min(capacity, len(pool)) + 1)
            for subset in itertools.combinations(pool, n)
        )

    return dp(0, ())


def reference(demands, writes, capacity):
    resident, result = set(), []
    for t, demand in enumerate(demands):
        resident -= writes[t]
        result.append(len(resident & demand))

        def future(token, t=t):
            for j in range(t + 1, len(demands)):
                if token in writes[j]:
                    break
                if token in demands[j]:
                    return j
            return len(demands)

        resident = set(
            sorted(resident | demand, key=lambda x: (future(x), x))[:capacity]
        )
    return result


def check(function, demands, writes, capacity):
    slots = [np.array(sorted(s) + [-1], dtype=np.int32).reshape(1, -1) for s in demands]
    written = [np.array(sorted(w), dtype=np.int32) for w in writes]
    values = pack(slots, written)
    actual = run(function, capacity, values)
    offsets = values[1][0]
    counts = [int(actual[a:b].sum()) for a, b in itertools.pairwise(offsets)]
    assert sum(counts) == exact(demands, writes, capacity), (
        demands,
        writes,
        capacity,
        counts,
    )
    assert counts == reference(demands, writes, capacity), (
        demands,
        writes,
        capacity,
        counts,
    )
    assert all(actual[int(i) - 1] == 0 for i in offsets[1:]), "Padding must miss"


def main():
    rng = random.Random(20261014)
    checked = 0
    with tempfile.TemporaryDirectory(prefix="horizonkv-belady-test-") as tmp:
        fn = compile_belady(tmp)
        # Exhaustive three-token, four-event traces, including empty batches.
        sets = [set(c) for n in range(4) for c in itertools.combinations(range(3), n)]
        for seq in itertools.product(sets, repeat=4):
            for capacity in range(4):
                check(fn, seq, [set()] * 4, capacity)
                checked += 1
        # Independent random resets/overwrites and new versions of old slots.
        for _ in range(300):
            n, t = rng.randrange(2, 6), rng.randrange(2, 9)
            demands = [
                set(rng.sample(range(n), rng.randrange(n + 1))) for _ in range(t)
            ]
            writes = [set(rng.sample(range(n), rng.randrange(n + 1))) for _ in range(t)]
            for capacity in range(n + 2):
                check(fn, demands, writes, capacity)
                checked += 1
        # Explicit invalidation fence: old x cannot earn new x's future hit.
        for capacity in (0, 1, 2, 5):
            check(fn, [{0, 1}, {1}, {0}, {0, 1}], [set(), {0}, set(), set()], capacity)
            checked += 1
        # A suffix outside the measured horizon must not influence retention.
        prefix = [{0, 1}, {0}, {1}]
        for capacity in (0, 1, 2):
            check(fn, prefix, [set()] * 3, capacity)
            checked += 1
        try:
            pack([np.array([[0, 0]], dtype=np.int32)], [np.array([], dtype=np.int32)])
        except AssertionError:
            pass
        else:
            raise AssertionError(
                "Duplicate identities are weighted demand; reject them"
            )
    print(
        f"PASS: {checked} compiled cases match exact DP and independent future scan; "
        "snapshot batches, bypass, padding, version fences, reset and finite prefix"
    )


if __name__ == "__main__":
    main()
