#include "vector_search/hnsw_graph.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace vector_search {

HnswGraph::HnswGraph(
    std::size_t dimension,
    std::vector<HnswNode> nodes,
    std::optional<NodeIndex> entry_node_index,
    GraphDegreeLimits degree_limits)
    : dimension_(dimension),
      nodes_(std::move(nodes)),
      entry_node_index_(entry_node_index),
      degree_limits_(degree_limits) {
    validate();
}

HnswGraph::HnswGraph(HnswGraph&& other) noexcept
    : dimension_(other.dimension_),
      nodes_(std::move(other.nodes_)),
      entry_node_index_(std::exchange(other.entry_node_index_, std::nullopt)),
      degree_limits_(other.degree_limits_) {
    // Leave the source as a valid empty graph.
    other.nodes_.clear();
}

HnswGraph& HnswGraph::operator=(HnswGraph other) noexcept {
    // Prepare the replacement before changing any owned data.
    using std::swap;
    swap(dimension_, other.dimension_);
    swap(nodes_, other.nodes_);
    swap(entry_node_index_, other.entry_node_index_);
    swap(degree_limits_, other.degree_limits_);
    return *this;
}

const HnswNode& HnswGraph::node(NodeIndex node_index) const {
    return nodes_.at(node_index);
}

std::span<const NodeIndex> HnswGraph::neighbors(NodeIndex node_index, std::size_t layer) const {
    return node(node_index).neighbors.at(layer);
}

void HnswGraph::validate() const {
    if (dimension_ == 0) {
        throw std::invalid_argument("graph dimension must be positive");
    }
    if (degree_limits_.base_max_degree == 0 || degree_limits_.upper_max_degree == 0) {
        throw std::invalid_argument("graph degree limits must be positive");
    }

    std::unordered_set<std::uint64_t> source_ids;
    source_ids.reserve(nodes_.size());
    std::size_t highest_level = 0;

    // Check all node shapes before following any links.
    for (const HnswNode& current_node : nodes_) {
        if (!source_ids.insert(current_node.point.id).second) {
            throw std::invalid_argument("graph source IDs must be distinct");
        }
        if (current_node.point.values.size() != dimension_) {
            throw std::invalid_argument("node dimension differs from graph dimension");
        }
        for (float coordinate : current_node.point.values) {
            if (!std::isfinite(coordinate)) {
                throw std::invalid_argument("node coordinates must be finite");
            }
        }
        // Subtract from a nonempty list count. Do not add to a possibly large level.
        if (current_node.neighbors.empty() ||
            current_node.level != current_node.neighbors.size() - 1) {
            throw std::invalid_argument("node level and layer list count differ");
        }
        highest_level = std::max(highest_level, current_node.level);
    }

    if (nodes_.empty()) {
        if (entry_node_index_) {
            throw std::invalid_argument("empty graph must not have an entry");
        }
        return;
    }
    if (!entry_node_index_ || *entry_node_index_ >= nodes_.size()) {
        throw std::invalid_argument("nonempty graph needs a valid entry");
    }
    if (nodes_[*entry_node_index_].level != highest_level) {
        throw std::invalid_argument("entry must be at the highest graph level");
    }

    for (NodeIndex node_index = 0; node_index < nodes_.size(); ++node_index) {
        const HnswNode& current_node = nodes_[node_index];
        for (std::size_t layer = 0; layer < current_node.neighbors.size(); ++layer) {
            const auto& links = current_node.neighbors[layer];
            const std::size_t degree_limit = layer == 0 ?
                degree_limits_.base_max_degree : degree_limits_.upper_max_degree;
            if (links.size() > degree_limit) {
                throw std::invalid_argument("neighbor count exceeds the layer degree limit");
            }

            std::unordered_set<NodeIndex> unique_neighbors;
            for (NodeIndex neighbor_index : links) {
                if (neighbor_index >= nodes_.size()) {
                    throw std::invalid_argument("neighbor index is outside the graph");
                }
                if (neighbor_index == node_index) {
                    throw std::invalid_argument("self-links are not allowed");
                }
                if (!unique_neighbors.insert(neighbor_index).second) {
                    throw std::invalid_argument("neighbor list contains a duplicate");
                }
                if (nodes_[neighbor_index].level < layer) {
                    throw std::invalid_argument("neighbor is absent from this layer");
                }
            }
        }
    }

    // A completed snapshot must expose every node from its layer-zero entry.
    std::vector<bool> reached(nodes_.size(), false);
    std::vector<NodeIndex> frontier{*entry_node_index_};
    reached[*entry_node_index_] = true;
    for (std::size_t frontier_index = 0; frontier_index < frontier.size(); ++frontier_index) {
        const NodeIndex node_index = frontier[frontier_index];
        for (NodeIndex neighbor_index : nodes_[node_index].neighbors[0]) {
            if (!reached[neighbor_index]) {
                reached[neighbor_index] = true;
                frontier.push_back(neighbor_index);
            }
        }
    }
    if (frontier.size() != nodes_.size()) {
        throw std::invalid_argument("node is unreachable from the layer-zero entry");
    }
}

}  // namespace vector_search
