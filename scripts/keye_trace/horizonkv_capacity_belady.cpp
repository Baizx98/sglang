// Equal-size, simultaneous-demand offline retention with optional admission.
// Each retained version spans an interval ending at its next demand. On overflow,
// reject the active interval with latest end; ties keep earlier insertions.
#include <algorithm>
#include <cstdint>
#include <vector>

extern "C" int capacity_belady(int32_t capacity, int32_t universe, int64_t events,
    const int64_t* offsets, const int32_t* slots,
    const int64_t* sorted_offsets, const int32_t* sorted_unique,
    const int64_t* write_offsets, const int32_t* writes, uint8_t* hits) {
    if (capacity < 0 || universe <= 0 || events < 0 || events > INT32_MAX - 1) return 1;
    const int32_t inf = static_cast<int32_t>(events);
    std::vector<int32_t> future(sorted_offsets[events], inf), seen(universe, inf);
    for (int64_t e = events; e-- > 0;) {
        // Reverse the forward write-before-demand order, ending old versions.
        for (int64_t i = sorted_offsets[e]; i < sorted_offsets[e + 1]; ++i) {
            int32_t x = sorted_unique[i];
            if (x < 0 || x >= universe) return 2;
            future[i] = seen[x]; seen[x] = static_cast<int32_t>(e);
        }
        for (int64_t i = write_offsets[e]; i < write_offsets[e + 1]; ++i) {
            int32_t x = writes[i];
            if (x < 0 || x >= universe) return 3;
            seen[x] = inf;
        }
    }
    // Intrusive time buckets give constant-time resident removal/insertion.
    std::vector<int32_t> head(events + 1, -1), left(universe, -1), right(universe, -1);
    std::vector<int32_t> end(universe, -1);
    int32_t size = 0, latest = -1;
    auto remove = [&](int32_t x) {
        int32_t t = end[x], a = left[x], b = right[x];
        if (a >= 0) right[a] = b; else head[t] = b;
        if (b >= 0) left[b] = a;
        end[x] = -1; --size;
    };
    auto insert = [&](int32_t x, int32_t t) {
        end[x] = t; left[x] = -1; right[x] = head[t];
        if (head[t] >= 0) left[head[t]] = x;
        head[t] = x; ++size; latest = std::max(latest, t);
    };
    for (int64_t e = 0; e < events; ++e) {
        for (int64_t i = write_offsets[e]; i < write_offsets[e + 1]; ++i) {
            int32_t x = writes[i];
            if (end[x] >= 0) remove(x);
        }
        for (int64_t i = offsets[e]; i < offsets[e + 1]; ++i) {
            int32_t x = slots[i];
            if (x < -1 || x >= universe) return 4;
            hits[i] = x >= 0 && end[x] >= 0;
            if (hits[i] && end[x] != e) return 5;
        }
        for (int64_t i = sorted_offsets[e]; i < sorted_offsets[e + 1]; ++i) {
            int32_t x = sorted_unique[i];
            if (end[x] >= 0) remove(x);
        }
        while (latest >= 0 && head[latest] < 0) --latest;
        for (int64_t i = sorted_offsets[e]; i < sorted_offsets[e + 1]; ++i) {
            int32_t x = sorted_unique[i], next = future[i];
            if (!capacity || next == inf) continue;
            if (next <= e) return 6;
            if (size == capacity) {
                if (next >= latest) continue;  // bypass; incumbent wins a tie
                remove(head[latest]);
                while (latest >= 0 && head[latest] < 0) --latest;
            }
            insert(x, next);
            if (size > capacity) return 7;
        }
    }
    return 0;
}
