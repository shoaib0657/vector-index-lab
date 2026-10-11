#include "vector_search/hnsw_insert.h"
#include "hnsw_insert_internal.h"
#include "hnsw_search_internal.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace vector_search {
namespace {

constexpr std::size_t maximum_level = 32;

void validate_insert_config(const HnswGraph& graph, Layer0InsertConfig config,
                            std::size_t level) {
    const auto limits = graph.degree_limits();
    if (config.neighbors_per_node == 0 ||
        config.ef_construction < config.neighbors_per_node || config.max_nodes == 0 ||
        config.neighbors_per_node > limits.base_max_degree ||
        (level > 0 && config.neighbors_per_node > limits.upper_max_degree) ||
        level > maximum_level) {
        throw std::invalid_argument("invalid insertion settings or level");
    }
}

void validate_point(const HnswGraph& graph, const Point& point, std::size_t max_nodes) {
    // Validate coordinates even when the graph is empty.
    if (point.values.size() != graph.dimension()) {
        throw std::invalid_argument("point and graph dimensions differ");
    }
    for (float coordinate : point.values) {
        if (!std::isfinite(coordinate)) {
            throw std::invalid_argument("coordinates must be finite");
        }
    }
    if (graph.size() >= max_nodes) {
        throw std::length_error("insertion node limit reached");
    }
    if (graph.size() >= std::vector<HnswNode>{}.max_size()) {
        throw std::length_error("insertion exceeds node vector limit");
    }
    for (NodeIndex node_index = 0; node_index < graph.size(); ++node_index) {
        const auto& node = graph.node(node_index);
        if (node.level > maximum_level) {
            throw std::invalid_argument("stored node level exceeds insertion limit");
        }
        if (node.point.id == point.id) {
            throw std::invalid_argument("source ID is already stored");
        }
    }
}

double squared_l2(std::span<const float> left, std::span<const float> right) {
    double distance = 0;
    for (std::size_t coordinate_index = 0; coordinate_index < left.size(); ++coordinate_index) {
        // Widen before subtraction to avoid float overflow and rounding.
        const double difference = static_cast<double>(left[coordinate_index]) -
                                  static_cast<double>(right[coordinate_index]);
        distance += difference * difference;
    }
    return distance;
}

void add_reverse_links(std::vector<HnswNode>& draft, NodeIndex new_node_index,
                       std::size_t layer, std::size_t degree_cap) {
    for (NodeIndex owner_index : draft[new_node_index].neighbors[layer]) {
        auto& neighbors = draft[owner_index].neighbors[layer];
        neighbors.push_back(new_node_index);
        if (neighbors.size() > degree_cap) {
            // Prune from this list's owner. Other lists keep their own links.
            const auto& owner_values = draft[owner_index].point.values;
            std::sort(neighbors.begin(), neighbors.end(), [&](NodeIndex left, NodeIndex right) {
                const auto& left_point = draft[left].point;
                const auto& right_point = draft[right].point;
                const double left_distance = squared_l2(owner_values, left_point.values);
                const double right_distance = squared_l2(owner_values, right_point.values);
                if (left_distance != right_distance) {
                    return left_distance < right_distance;
                }
                return left_point.id < right_point.id;
            });
            neighbors.resize(degree_cap);
        }
    }
}

}  // namespace

std::size_t detail::level_from_word(std::uint64_t word, std::size_t neighbors_per_node,
                                    std::size_t level_cap) {
    static_assert(std::numeric_limits<double>::is_iec559 &&
                  std::numeric_limits<double>::radix == 2 &&
                  std::numeric_limits<double>::digits == 53);
    if (neighbors_per_node < 2 || level_cap > maximum_level) {
        throw std::invalid_argument("invalid level settings");
    }
    // Add one so the logarithm never receives zero.
    const double uniform = static_cast<double>((word >> 11) + 1) * 0x1.0p-53;
    const double raw_level = std::floor(-std::log(uniform) /
                                       std::log(static_cast<double>(neighbors_per_node)));
    // Bound the floating value before converting it to a node level.
    return static_cast<std::size_t>(std::min(raw_level, static_cast<double>(level_cap)));
}

HnswGraph detail::insert_at_level(const HnswGraph& graph, Point point,
                                 Layer0InsertConfig config, std::size_t level) {
    validate_insert_config(graph, config, level);
    validate_point(graph, point, config.max_nodes);
    const auto limits = graph.degree_limits();

    // Keep all edits local until the completed snapshot passes validation.
    std::vector<HnswNode> draft;
    draft.reserve(graph.size() + 1);
    for (NodeIndex node_index = 0; node_index < graph.size(); ++node_index) {
        draft.push_back(graph.node(node_index));
    }
    const NodeIndex new_node_index = draft.size();
    draft.push_back({std::move(point), level, std::vector<std::vector<NodeIndex>>(level + 1)});
    if (graph.size() == 0) {
        return HnswGraph(graph.dimension(), std::move(draft), new_node_index, limits);
    }

    const NodeIndex old_entry = graph.entry_node_index().value();
    const std::size_t old_top_level = graph.node(old_entry).level;
    const auto& query = draft[new_node_index].point.values;
    std::vector<NodeIndex> entries{old_entry};
    for (std::size_t layer = old_top_level; layer > level; --layer) {
        const auto retained = detail::search_layer(graph, query, entries, layer, 1);
        entries[0] = retained.front().node_index;
    }
    for (std::size_t layer = std::min(level, old_top_level);; --layer) {
        // Search the old snapshot. The new node cannot select itself.
        const auto retained = detail::search_layer(graph, query, entries, layer, config.ef_construction);
        auto& selected = draft[new_node_index].neighbors[layer];
        const auto count = std::min(config.neighbors_per_node, retained.size());
        selected.reserve(count);
        for (std::size_t candidate_index = 0; candidate_index < count; ++candidate_index) {
            selected.push_back(retained[candidate_index].node_index);
        }
        add_reverse_links(draft, new_node_index, layer,
                          layer == 0 ? limits.base_max_degree : limits.upper_max_degree);
        if (layer == 0) {
            break;
        }
        // Carry all W entries down. Selection and pruning do not shorten W.
        entries.clear();
        entries.reserve(retained.size());
        for (const auto& candidate : retained) {
            entries.push_back(candidate.node_index);
        }
    }
    const NodeIndex entry = level > old_top_level ? new_node_index : old_entry;
    // The constructor also checks directed reachability from the entry.
    return HnswGraph(graph.dimension(), std::move(draft), entry, limits);
}

HnswGraph insert_layer0(const HnswGraph& graph, Point point, Layer0InsertConfig config) {
    for (NodeIndex node_index = 0; node_index < graph.size(); ++node_index) {
        if (graph.node(node_index).level != 0) {
            throw std::invalid_argument("layer-zero insertion requires level-zero nodes");
        }
    }
    return detail::insert_at_level(graph, std::move(point), config, 0);
}

HnswBuildState::HnswBuildState(std::size_t dimension, GraphDegreeLimits limits,
                               HnswBuildConfig config)
    : graph_(dimension, {}, std::nullopt, limits), config_(config), generator_(config.seed) {
    if (config.neighbors_per_node < 2 || config.level_cap > maximum_level ||
        config.neighbors_per_node > limits.upper_max_degree) {
        throw std::invalid_argument("invalid seeded build settings");
    }
    validate_insert_config(graph_,
        {config.neighbors_per_node, config.ef_construction, config.max_nodes}, config.level_cap);
}

HnswBuildState::HnswBuildState(HnswGraph graph, HnswBuildConfig config,
                               std::mt19937_64 generator)
    : graph_(std::move(graph)), config_(config), generator_(std::move(generator)) {}

HnswBuildState& HnswBuildState::operator=(HnswBuildState other) noexcept {
    static_assert(std::is_nothrow_move_constructible_v<std::mt19937_64> &&
                  std::is_nothrow_move_assignable_v<std::mt19937_64>);
    // The value parameter prepares any throwing copy before assignment starts.
    graph_ = std::move(other.graph_);
    config_ = other.config_;
    generator_ = std::move(other.generator_);
    return *this;
}

HnswBuildState HnswBuildState::insert(Point point) const {
    validate_point(graph_, point, config_.max_nodes);
    // A rejected draft must preserve the next draw in this state.
    auto generator = generator_;
    const auto level = detail::level_from_word(generator(), config_.neighbors_per_node, config_.level_cap);
    auto graph = detail::insert_at_level(graph_, std::move(point),
        {config_.neighbors_per_node, config_.ef_construction, config_.max_nodes}, level);
    return HnswBuildState(std::move(graph), config_, std::move(generator));
}

}  // namespace vector_search
