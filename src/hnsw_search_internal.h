#pragma once

#include "vector_search/hnsw_search.h"

namespace vector_search::detail {

struct ScoredNode {
    NodeIndex node_index;
    std::uint64_t source_id;
    double squared_distance;
};

// Start with all entries and return the full retained set in sorted order.
// The public trace schema can describe only a single entry.
std::vector<ScoredNode> search_layer(
    const HnswGraph& graph, std::span<const float> query,
    std::span<const NodeIndex> entries, std::size_t layer, std::size_t ef,
    LayerSearchTrace* trace = nullptr);

}  // namespace vector_search::detail
