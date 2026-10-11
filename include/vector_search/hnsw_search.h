#pragma once

#include "vector_search/hnsw_graph.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace vector_search {

struct LayerSearchTrace {
    std::size_t layer;
    std::uint64_t entry_source_id;
    // Record discovery order, including rejected neighbors.
    std::vector<std::uint64_t> visited_source_ids;
    std::vector<std::uint64_t> expanded_source_ids;
    std::vector<Neighbor> retained_neighbors;
    std::optional<Neighbor> stopped_candidate;
};

struct HnswSearchTrace {
    std::vector<LayerSearchTrace> layers;
};

// Read the graph with query-local heaps and visited flags.
// Return results in distance order. Use source IDs to order equal distances.
// For a nonempty graph and positive k, require a finite matching query and ef >= k.
// If k is zero or the graph is empty, return before checking those inputs.
// Replace a requested trace after success. Leave it unchanged on an exception.
std::vector<Neighbor> hnsw_top_k(
    const HnswGraph& graph,
    std::span<const float> query,
    std::size_t k,
    std::size_t ef_search,
    HnswSearchTrace* trace = nullptr);

}  // namespace vector_search
