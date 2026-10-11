#pragma once

#include "vector_search/hnsw_graph.h"

#include <cstddef>
#include <cstdint>
#include <random>

namespace vector_search {

struct Layer0InsertConfig {
    std::size_t neighbors_per_node;
    std::size_t ef_construction;
    std::size_t max_nodes;
};

// Return a new owned snapshot. The input graph stays unchanged on failure.
// Every node has level zero. Prune each old list from its owner's vector.
// Invalid input or an unreachable draft throws invalid_argument.
// A full node limit throws length_error. Allocation failures propagate.
// Point is passed by value. A moved argument can be consumed on failure.
HnswGraph insert_layer0(const HnswGraph& graph, Point point, Layer0InsertConfig config);

struct HnswBuildConfig {
    std::size_t neighbors_per_node;
    std::size_t ef_construction;
    std::size_t max_nodes;
    std::size_t level_cap;
    std::uint64_t seed;
};

// Own the completed graph and its next random draw together.
// A const insertion returns a new state. Failure preserves this state.
// Copying a state forks its construction history.
class HnswBuildState {
public:
    HnswBuildState(std::size_t dimension, GraphDegreeLimits limits, HnswBuildConfig config);
    HnswBuildState(const HnswBuildState&) = default;
    HnswBuildState(HnswBuildState&&) noexcept = default;
    HnswBuildState& operator=(HnswBuildState other) noexcept;

    const HnswGraph& graph() const noexcept { return graph_; }
    HnswBuildConfig config() const noexcept { return config_; }
    HnswBuildState insert(Point point) const;

private:
    HnswBuildState(HnswGraph graph, HnswBuildConfig config, std::mt19937_64 generator);

    HnswGraph graph_;
    HnswBuildConfig config_;
    std::mt19937_64 generator_;
};

}  // namespace vector_search
