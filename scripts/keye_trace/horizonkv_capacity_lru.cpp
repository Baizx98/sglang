// Batch-snapshot LRU matching HostLayer.stage, including write invalidation.
#include <algorithm>
#include <cstdint>
#include <vector>

static int replay_impl(int32_t capacity, int32_t universe, int64_t events,
    const int64_t* offsets, const int32_t* slots,
    const int64_t* sorted_offsets, const int32_t* sorted_unique,
    const int64_t* write_offsets, const int32_t* writes, uint8_t* hits,
    bool promote_hits) {
    if (capacity < 0 || universe <= 0) return 1;
    std::vector<int32_t> before(universe, -2), after(universe, -2);
    std::vector<int64_t> stamp(universe, -1);
    int32_t oldest = -1, newest = -1, size = 0;
    auto remove = [&](int32_t token) {
        int32_t left = before[token], right = after[token];
        if (left >= 0) after[left] = right; else oldest = right;
        if (right >= 0) before[right] = left; else newest = left;
        before[token] = after[token] = -2;
        --size;
    };
    auto append = [&](int32_t token) {
        before[token] = newest; after[token] = -1;
        if (newest >= 0) after[newest] = token; else oldest = token;
        newest = token; ++size;
    };
    for (int64_t e = 0; e < events; ++e) {
        for (int64_t i = write_offsets[e]; i < write_offsets[e + 1]; ++i) {
            int32_t token = writes[i];
            if (token < 0 || token >= universe) return 2;
            if (before[token] != -2) remove(token);
        }
        for (int64_t i = offsets[e]; i < offsets[e + 1]; ++i) {
            int32_t token = slots[i];
            if (token >= universe || token < -1) return 3;
            hits[i] = token >= 0 && before[token] != -2;
        }
        if (!capacity) continue;
        for (int64_t i = sorted_offsets[e]; i < sorted_offsets[e + 1]; ++i) {
            int32_t token = sorted_unique[i];
            if (token < 0 || token >= universe) return 4;
            if (before[token] != -2) {
                stamp[token] = e;
                remove(token); append(token);
            }
        }
        for (int64_t i = sorted_offsets[e]; i < sorted_offsets[e + 1]; ++i) {
            int32_t token = sorted_unique[i];
            if (stamp[token] == e) continue;
            if (size == capacity) remove(oldest);
            append(token);
        }
        if (promote_hits) {
            for (int64_t i = sorted_offsets[e]; i < sorted_offsets[e + 1]; ++i) {
                int32_t token = sorted_unique[i];
                if (stamp[token] == e && before[token] != -2) {
                    remove(token); append(token);
                }
            }
        }
    }
    return 0;
}

extern "C" int capacity_replay(int32_t c, int32_t u, int64_t e,
    const int64_t* o, const int32_t* s, const int64_t* so, const int32_t* su,
    const int64_t* wo, const int32_t* w, uint8_t* h) {
    return replay_impl(c, u, e, o, s, so, su, wo, w, h, false);
}

extern "C" int capacity_replay_refined(int32_t c, int32_t u, int64_t e,
    const int64_t* o, const int32_t* s, const int64_t* so, const int32_t* su,
    const int64_t* wo, const int32_t* w, uint8_t* h) {
    return replay_impl(c, u, e, o, s, so, su, wo, w, h, true);
}
