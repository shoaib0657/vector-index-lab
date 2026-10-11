#pragma once

#include "vector_search/exact_search.h"

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace vector_search {

// A node index names a position in the graph's node array.
using NodeIndex = std::size_t;

struct HnswNode {
    Point point;
    std::size_t level;

    // Keep one list for each layer from zero through level.
    // Links name node indices in the graph's node array.
    std::vector<std::vector<NodeIndex>> neighbors;
};

struct GraphDegreeLimits {
    std::size_t base_max_degree;
    std::size_t upper_max_degree;
};

// Own a completed static graph. Construction rejects invalid snapshots.
// Reads return const views. Keep the graph alive and its storage unchanged
// while using a view. Acquire new views after moving or assigning the graph.
class HnswGraph {
public:
    HnswGraph(
        std::size_t dimension,
        std::vector<HnswNode> nodes,
        std::optional<NodeIndex> entry_node_index,
        GraphDegreeLimits degree_limits);

    HnswGraph(const HnswGraph&) = default;
    HnswGraph(HnswGraph&& other) noexcept;
    HnswGraph& operator=(HnswGraph other) noexcept;

    std::size_t size() const noexcept { return nodes_.size(); }
    std::size_t dimension() const noexcept { return dimension_; }
    std::optional<NodeIndex> entry_node_index() const noexcept {
        return entry_node_index_;
    }

    // Invalid node indices and absent layers cause an out-of-range exception.
    const HnswNode& node(NodeIndex node_index) const;
    std::span<const NodeIndex> neighbors(NodeIndex node_index, std::size_t layer) const;

private:
    void validate() const;

    std::size_t dimension_;
    std::vector<HnswNode> nodes_;
    std::optional<NodeIndex> entry_node_index_;
    GraphDegreeLimits degree_limits_;
};

}  // namespace vector_search
