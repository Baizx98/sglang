// Two-tier replay: fixed prefetch requests, exact causal demand, persistent HBM.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

struct Step {
    std::vector<int32_t> predicted, prefetch, demand, writes, used_current;
};

struct Row {
    int last_use = -1;
    int previous_use = -1;
    int last_prefetch = -1;
    int arrival = -1;
    int last_step = -1;
    double heat = 0;
    double prediction = 0;
    bool resident = false;
    bool pending = false;
    bool evicted = false;
    bool downloaded = false;
};

struct Counts {
    int64_t requests = 0, demand_loads = 0, prefetch_loads = 0;
    int64_t repeated_h2d = 0, reload_after_eviction = 0, evictions = 0;
    int64_t prefetched_consumed = 0, unused_evicted = 0, unused_at_end = 0;
    int64_t displaced_demand = 0, rescued_demand = 0;
};

struct Policy {
    std::string name;
    double half_life;
    double hint_bonus;
    double prior_bonus;
};

class Replay {
    using Key = std::tuple<double, int, int>;
    std::vector<Row> rows;
    std::set<Key> order;
    int capacity, now = 0, stamp = 0;
    const Policy policy;
    const bool lfu, predicted, issued_only, frequent_pair, touch_prefetch, pure_prediction;
    const std::vector<float>& prior;
    std::vector<double> decay;
    Counts stats;

    double frequency(const Row& row) const {
        if (row.last_step < 0) return 0;
        return row.heat * decay[now - row.last_step];
    }

    Key rank(int id) const {
        const auto& r = rows[id];
        if (pure_prediction) return {r.prediction, r.arrival, id};
        if (policy.name == "turn" || policy.name == "role") return {prior.at(id), r.arrival, id};
        const int last = touch_prefetch ? std::max(r.last_use, r.last_prefetch) : r.last_use;
        double value;
        if (lfu) value = frequency(r);
        else if (frequent_pair) value = r.previous_use;
        else value = last;
        if (predicted) {
            if (!lfu) value = last < 0 ? now - 16 : r.last_step;
            value += policy.hint_bonus * r.prediction;
        }
        if (policy.prior_bonus != 0) value += policy.prior_bonus * prior.at(id);
        return {value, last < 0 ? r.arrival : last, id};
    }

    void erase(int id) {
        if (rows[id].resident) order.erase(rank(id));
    }

    void evict() {
        const int id = std::get<2>(*order.begin());
        auto& r = rows[id];
        if (r.pending) ++stats.unused_evicted;
        r.pending = false;
        r.resident = false;
        r.evicted = true;
        ++stats.evictions;
        order.erase(order.begin());
    }

    void insert(int id, bool prefetch) {
        auto& r = rows[id];
        if (r.resident || capacity == 0) return;
        if (static_cast<int>(order.size()) == capacity) evict();
        r.pending = prefetch;
        r.resident = true;
        r.arrival = stamp++;
        order.insert(rank(id));
    }

    void transfer(int id, bool prefetch) {
        auto& r = rows[id];
        if (prefetch) ++stats.prefetch_loads;
        else ++stats.demand_loads;
        if (r.downloaded) ++stats.repeated_h2d;
        if (r.evicted) ++stats.reload_after_eviction;
        r.downloaded = true;
    }

    void consume(int id) {
        auto& r = rows[id];
        erase(id);
        if (r.pending) {
            ++stats.prefetched_consumed;
            r.pending = false;
        }
        r.heat = frequency(r) + 1;
        r.last_step = now;
        r.previous_use = r.last_use;
        r.last_use = stamp++;
        if (r.resident) order.insert(rank(id));
        else insert(id, false);
    }

    void begin_step(const Step& step, const Step* previous) {
        if (!predicted && !lfu) return;
        if (predicted && previous) {
            for (int id : (issued_only ? previous->prefetch : previous->predicted)) rows[id].prediction = 0;
        }
        if (predicted) {
            const auto& hints = issued_only ? step.prefetch : step.predicted;
            const double n = hints.size();
            for (size_t i = 0; i < hints.size(); ++i) {
                rows[hints[i]].prediction = issued_only ? 1 : 1 - 0.5 * i / n;
            }
        }
        std::vector<int> ids;
        ids.reserve(order.size());
        for (const auto& k : order) ids.push_back(std::get<2>(k));
        order.clear();
        for (int id : ids) order.insert(rank(id));
    }

    void prefetch(int id) {
        if (capacity == 0) return;
        auto& r = rows[id];
        if (!r.resident) {
            transfer(id, true);
            insert(id, true);
        }
        if (touch_prefetch && r.resident) {
            erase(id);
            r.last_prefetch = stamp++;
            order.insert(rank(id));
        }
    }

public:
    Replay(int count, int budget, int steps, Policy p, const std::vector<float>& values)
        : rows(count), capacity(budget), policy(std::move(p)),
          lfu(policy.name.rfind("lfu", 0) == 0),
          predicted(policy.name.find("pred") != std::string::npos || policy.name.find("issued") != std::string::npos),
          issued_only(policy.name.find("issued") != std::string::npos),
          frequent_pair(policy.name == "lru2"), touch_prefetch(policy.name == "lru-touch"),
          pure_prediction(policy.name == "pred"), prior(values), decay(steps, 1) {
        if (lfu) {
            for (int i = 1; i < steps; ++i) decay[i] = std::exp2(-i / policy.half_life);
        }
    }

    Counts run(const std::vector<Step>& steps, int limit) {
        for (int i_step = 0; i_step < limit; ++i_step) {
            const auto& step = steps[i_step];
            begin_step(step, now > 0 ? &steps[now - 1] : nullptr);
            std::vector<bool> before;
            before.reserve(step.demand.size());
            for (int id : step.demand) before.push_back(rows[id].resident);
            for (int id : step.prefetch) prefetch(id);

            // The entire demand batch sees one snapshot after issued prefetch.
            // Shared transient staging serves misses; persistent admission follows.
            for (size_t i = 0; i < step.demand.size(); ++i) {
                const int id = step.demand[i];
                const bool after = rows[id].resident;
                ++stats.requests;
                if (!after) transfer(id, false);
                if (before[i] && !after) ++stats.displaced_demand;
                if (!before[i] && after) ++stats.rescued_demand;
                if (rows[id].pending) {
                    ++stats.prefetched_consumed;
                    rows[id].pending = false;
                }
            }
            for (int id : step.demand) consume(id);
            for (int id : step.writes) insert(id, false);
            for (int id : step.used_current) consume(id);
            ++now;
        }
        for (const auto& k : order) {
            if (rows[std::get<2>(k)].pending) ++stats.unused_at_end;
        }
        if (capacity && stats.prefetch_loads != stats.prefetched_consumed + stats.unused_evicted + stats.unused_at_end) {
            throw std::logic_error("unaccounted prefetch transfer");
        }
        return stats;
    }
};

int32_t read_number(std::istream& in) {
    int32_t n = 0;
    if (!in.read(reinterpret_cast<char*>(&n), sizeof(n))) throw std::runtime_error("truncated input");
    return n;
}

std::vector<int32_t> read_ids(std::istream& in, int count) {
    int n = read_number(in);
    if (n < 0 || n > count) throw std::runtime_error("invalid group size");
    std::vector<int32_t> ids(n);
    if (!in.read(reinterpret_cast<char*>(ids.data()), n * sizeof(int32_t))) {
        throw std::runtime_error("truncated IDs");
    }
    for (int id : ids) {
        if (id < 0 || id >= count) throw std::runtime_error("block ID out of range");
    }
    return ids;
}

std::vector<float> read_prior(std::istream& in, int count) {
    int n = read_number(in);
    if (n != 0 && n != count) throw std::runtime_error("invalid prior width");
    std::vector<float> prior(n);
    if (n && !in.read(reinterpret_cast<char*>(prior.data()), n * sizeof(float))) {
        throw std::runtime_error("truncated prior");
    }
    for (float score : prior) {
        if (!(score >= 0 && score <= 1)) throw std::runtime_error("invalid prior score");
    }
    return prior;
}

int main() {
    try {
        std::ios::sync_with_stdio(false);
        int count = read_number(std::cin);
        int num_steps = read_number(std::cin);
        if (count <= 0 || num_steps <= 0) throw std::runtime_error("empty input");
        const auto turn = read_prior(std::cin, count);
        const auto role = read_prior(std::cin, count);
        std::vector<Step> steps(num_steps);
        for (auto& step : steps) {
            step.predicted = read_ids(std::cin, count);
            step.prefetch = read_ids(std::cin, count);
            step.demand = read_ids(std::cin, count);
            step.writes = read_ids(std::cin, count);
            step.used_current = read_ids(std::cin, count);
        }
        int capacity, limit;
        Policy policy;
        while (std::cin >> capacity >> policy.name >> policy.half_life >> policy.hint_bonus >> policy.prior_bonus >> limit) {
            if (limit == 0) limit = num_steps;
            if (capacity < 0 || policy.half_life <= 0 || limit < 1 || limit > num_steps) {
                throw std::runtime_error("invalid policy parameters");
            }
            const auto start = std::chrono::steady_clock::now();
            const auto& prior = policy.name.find("role") != std::string::npos ? role : turn;
            auto c = Replay(count, capacity, num_steps, policy, prior).run(steps, limit);
            const auto elapsed = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            std::cout << c.requests << ' ' << c.demand_loads << ' ' << c.prefetch_loads << ' '
                      << c.repeated_h2d << ' ' << c.reload_after_eviction << ' ' << c.evictions << ' '
                      << c.prefetched_consumed << ' ' << c.unused_evicted << ' ' << c.unused_at_end << ' '
                      << c.displaced_demand << ' ' << c.rescued_demand << ' '
                      << sizeof(Row) * static_cast<int64_t>(count) << ' ' << elapsed << '\n';
        }
    } catch (const std::exception& problem) {
        std::cerr << problem.what() << '\n';
        return 1;
    }
}
