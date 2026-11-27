// Nonpreemptive, one-request-per-GPU event replay kernel for RcLLM+ TPDS cluster evaluation.
// Each replica maintains an ordered queue of first-token completion events; a slot becomes free at TTFT.
#include <algorithm>
#include <cstdint>
#include <deque>
#include <vector>

extern "C" void replay(int n, int replicas, int route, double kv_tokens,
                       const int64_t* requests, const double* arrivals,
                       const int64_t* tokens, const double* service, const double* hit,
                       double* start, double* finish, int64_t* assigned) {
    struct Job {
        double finish;
        int64_t tokens;
    };
    std::vector<std::deque<Job>> queues(replicas);
    std::vector<int> backlog(replicas);
    for (int i = 0; i < n; ++i) {
        const auto row = requests[i];
        const auto time = arrivals[i];
        int replica = i % replicas;  // Round-robin routing (route == 0).
        if (route != 0) {
            int max_backlog = 0;
            for (int r = 0; r < replicas; ++r) {
                auto& queue = queues[r];
                while (!queue.empty() && queue.front().finish <= time) {
                    queue.pop_front();
                }
                backlog[r] = static_cast<int>(queue.size());
                max_backlog = std::max(max_backlog, backlog[r]);
            }
            double best_score = -1.0;
            for (int r = 0; r < replicas; ++r) {
                const double match = hit[row * replicas + r];
                const double occupancy = queues[r].empty() ? 0.0 : queues[r].front().tokens / kv_tokens;
                const double relative_queue = max_backlog ? static_cast<double>(backlog[r]) / max_backlog : 0.0;
                const double idle = 1.0 - 0.5 * (relative_queue + occupancy);
                const double score = route == 1 ? 0.5 * (match + idle) : (route == 2 ? match : idle);
                if (score > best_score) {
                    best_score = score;
                    replica = r;
                }
            }
        }
        auto& queue = queues[replica];
        if (route == 0) {
            while (!queue.empty() && queue.front().finish <= time) {
                queue.pop_front();
            }
        }
        assigned[i] = replica;
        start[i] = queue.empty() ? time : queue.back().finish;
        finish[i] = start[i] + service[row * replicas + replica];
        queue.push_back({finish[i], tokens[row]});
    }
}
