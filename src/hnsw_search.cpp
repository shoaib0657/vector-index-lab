#include "vector_search/hnsw_search.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>
#include <utility>

namespace vector_search {
namespace {

struct ScoredNode {
    NodeIndex node_index;
    std::uint64_t source_id;
    double squared_distance;
};

bool nearer(const ScoredNode& left, const ScoredNode& right) {
    if (left.squared_distance != right.squared_distance) {
        return left.squared_distance < right.squared_distance;
    }
    return left.source_id < right.source_id;
}

struct CandidatePriority {
    bool operator()(const ScoredNode& left, const ScoredNode& right) const {
        // Give nearer candidates higher priority.
        return nearer(right, left);
    }
};

struct BestPriority {
    bool operator()(const ScoredNode& left, const ScoredNode& right) const {
        // Expose the farthest retained node for eviction.
        return nearer(left, right);
    }
};

ScoredNode score_node(
    const HnswGraph& graph, std::span<const float> query, NodeIndex node_index) {
    const auto& point = graph.node(node_index).point;
    double squared_distance = 0.0;
    for (std::size_t coordinate_index = 0; coordinate_index < query.size(); ++coordinate_index) {
        // Widen both values before subtraction to avoid float overflow.
        const double difference = static_cast<double>(query[coordinate_index]) -
            static_cast<double>(point.values[coordinate_index]);
        squared_distance += difference * difference;
    }
    return {node_index, point.id, squared_distance};
}

std::vector<ScoredNode> search_layer(
    const HnswGraph& graph,
    std::span<const float> query,
    NodeIndex entry_node_index,
    std::size_t layer,
    std::size_t ef,
    LayerSearchTrace* trace) {
    // Each layer call starts with new query state.
    std::priority_queue<ScoredNode, std::vector<ScoredNode>, CandidatePriority> candidates;
    std::priority_queue<ScoredNode, std::vector<ScoredNode>, BestPriority> best;
    std::vector<std::uint8_t> visited(graph.size(), 0);
    const auto entry = score_node(graph, query, entry_node_index);
    candidates.push(entry);
    best.push(entry);
    visited[entry_node_index] = 1;
    if (trace) {
        trace->layer = layer;
        trace->entry_source_id = entry.source_id;
        trace->visited_source_ids.push_back(entry.source_id);
    }

    while (!candidates.empty()) {
        const ScoredNode current = candidates.top();
        candidates.pop();
        // Equal distances still permit expansion.
        if (current.squared_distance > best.top().squared_distance) {
            if (trace) {
                trace->stopped_candidate = Neighbor{current.source_id, current.squared_distance};
            }
            break;
        }
        if (trace) {
            trace->expanded_source_ids.push_back(current.source_id);
        }
        for (NodeIndex neighbor_index : graph.neighbors(current.node_index, layer)) {
            if (visited[neighbor_index]) {
                continue;
            }
            // Mark rejected neighbors as visited too.
            visited[neighbor_index] = 1;
            const auto neighbor = score_node(graph, query, neighbor_index);
            if (trace) {
                trace->visited_source_ids.push_back(neighbor.source_id);
            }
            // Read the current threshold for each neighbor.
            if (best.size() < ef || neighbor.squared_distance < best.top().squared_distance) {
                candidates.push(neighbor);
                best.push(neighbor);
                if (best.size() > ef) {
                    // An evicted node can still remain in the candidate heap.
                    best.pop();
                }
            }
        }
    }

    std::vector<ScoredNode> retained;
    retained.reserve(best.size());
    while (!best.empty()) {
        retained.push_back(best.top());
        best.pop();
    }
    // Sort before choosing the next entry or final results.
    std::sort(retained.begin(), retained.end(), nearer);
    if (trace) {
        trace->retained_neighbors.reserve(retained.size());
        for (const auto& node : retained) {
            trace->retained_neighbors.push_back({node.source_id, node.squared_distance});
        }
    }
    return retained;
}

}  // namespace

std::vector<Neighbor> hnsw_top_k(
    const HnswGraph& graph,
    std::span<const float> query,
    std::size_t k,
    std::size_t ef_search,
    HnswSearchTrace* trace) {
    if (k == 0 || graph.size() == 0) {
        if (trace) {
            trace->layers.clear();
        }
        return {};
    }
    if (query.size() != graph.dimension()) {
        throw std::invalid_argument("query dimension differs from graph dimension");
    }
    for (float coordinate : query) {
        if (!std::isfinite(coordinate)) {
            throw std::invalid_argument("query coordinates must be finite");
        }
    }
    if (ef_search < k) {
        throw std::invalid_argument("ef_search must be at least k");
    }

    // Keep the caller's trace unchanged until all query work succeeds.
    HnswSearchTrace prepared_trace;
    const auto run_layer = [&](NodeIndex entry_node_index, std::size_t layer, std::size_t ef) {
        LayerSearchTrace layer_trace{};
        auto retained = search_layer(
            graph, query, entry_node_index, layer, ef, trace ? &layer_trace : nullptr);
        if (trace) {
            prepared_trace.layers.push_back(std::move(layer_trace));
        }
        return retained;
    };

    NodeIndex entry_node_index = graph.entry_node_index().value();
    for (std::size_t layer = graph.node(entry_node_index).level; layer > 0; --layer) {
        const auto retained = run_layer(entry_node_index, layer, 1);
        entry_node_index = retained.front().node_index;
    }
    if (ef_search >= graph.size()) {
        // The stored entry reaches every node through layer-zero links.
        // Use it for a full-width search, including directed graphs.
        entry_node_index = graph.entry_node_index().value();
    }
    auto retained = run_layer(entry_node_index, 0, ef_search);
    retained.resize(std::min(k, retained.size()));
    std::vector<Neighbor> results;
    results.reserve(retained.size());
    for (const auto& node : retained) {
        results.push_back({node.source_id, node.squared_distance});
    }
    if (trace) {
        *trace = std::move(prepared_trace);
    }
    return results;
}

}  // namespace vector_search
