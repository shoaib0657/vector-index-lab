#include "vector_search/hnsw_insert.h"
#include "vector_search/hnsw_search.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using vector_search::HnswGraph;
using vector_search::HnswNode;
using vector_search::Layer0InsertConfig;
using vector_search::Neighbor;
using vector_search::NodeIndex;
using vector_search::Point;
using vector_search::hnsw_top_k;
using vector_search::insert_layer0;

constexpr Layer0InsertConfig config{1, 3, 3};
const std::array<float, 2> query{0, 0};

HnswGraph empty_graph() {
    return HnswGraph(2, {}, std::nullopt, {1, 1});
}

// These completed lists are independent of the insertion implementation.
HnswGraph pair_graph() {
    return HnswGraph(2,
        {{{10, {0, 0}}, 0, {{1}}}, {{20, {4, 0}}, 0, {{0}}}}, 0, {1, 1});
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_links(const HnswGraph& graph, NodeIndex node_index,
                   std::initializer_list<NodeIndex> expected) {
    require(std::ranges::equal(graph.neighbors(node_index, 0), expected),
            "base neighbor indices or order differ");
}

bool same_results(std::span<const Neighbor> left, std::span<const Neighbor> right) {
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t result_index = 0; result_index < left.size(); ++result_index) {
        if (left[result_index].id != right[result_index].id ||
            left[result_index].squared_distance != right[result_index].squared_distance) {
            return false;
        }
    }
    return true;
}

bool same_graph(const HnswGraph& left, const HnswGraph& right) {
    if (left.size() != right.size() || left.dimension() != right.dimension() ||
        left.entry_node_index() != right.entry_node_index() ||
        left.degree_limits().base_max_degree != right.degree_limits().base_max_degree ||
        left.degree_limits().upper_max_degree != right.degree_limits().upper_max_degree) {
        return false;
    }
    for (NodeIndex node_index = 0; node_index < left.size(); ++node_index) {
        const auto& a = left.node(node_index);
        const auto& b = right.node(node_index);
        if (a.point.id != b.point.id || a.point.values != b.point.values ||
            a.level != b.level || a.neighbors != b.neighbors) {
            return false;
        }
    }
    return true;
}

template <class Exception>
void require_rejection(const HnswGraph& graph, Point point, Layer0InsertConfig settings) {
    const HnswGraph before = graph;
    const auto old_results = hnsw_top_k(graph, query, 2, 3);
    const HnswNode* old_node = graph.size() ? &graph.node(0) : nullptr;
    const auto old_links = graph.size() ? graph.neighbors(0, 0) : std::span<const NodeIndex>{};
    bool rejected = false;
    try {
        (void)insert_layer0(graph, std::move(point), settings);
    } catch (const Exception&) {
        rejected = true;
    }
    require(rejected, "insertion did not throw the required exception type");
    require(same_graph(graph, before), "rejected insertion changed graph data or metadata");
    require(same_results(hnsw_top_k(graph, query, 2, 3), old_results),
            "rejected insertion changed query results");
    if (old_node) {
        require(old_node == &graph.node(0) && old_links.data() == graph.neighbors(0, 0).data() &&
                old_node->point.values == before.node(0).point.values &&
                std::ranges::equal(old_links, before.neighbors(0, 0)),
                "rejected insertion invalidated existing graph views");
    }
}

HnswGraph three_inserts() {
    auto graph = empty_graph();
    graph = insert_layer0(graph, {10, {0, 0}}, config);
    graph = insert_layer0(graph, {20, {4, 0}}, config);
    return insert_layer0(graph, {30, {5, 0}}, config);
}

}  // namespace

int main() {
    int cases = 0;
    int failures = 0;
    const auto run = [&](std::string_view name, auto check) {
        ++cases;
        try {
            check();
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        }
    };

    run("first insert creates an owned level-zero entry", [] {
        const auto input = empty_graph();
        const auto graph = insert_layer0(input, {10, {0, 0}}, config);
        require(graph.size() == 1 && graph.entry_node_index() == 0 &&
                graph.node(0).point.id == 10 && graph.node(0).point.values == std::vector<float>{0, 0} &&
                graph.node(0).level == 0 && graph.node(0).neighbors.size() == 1,
                "first point or entry differs");
        require_links(graph, 0, {});
        require(input.size() == 0 && !input.entry_node_index(), "empty input was changed");
    });
    run("second insert appends without changing the first node index", [] {
        const auto first = insert_layer0(empty_graph(), {10, {0, 0}}, config);
        const auto graph = insert_layer0(first, {20, {4, 0}}, config);
        require(graph.size() == 2 && graph.entry_node_index() == 0 &&
                graph.node(0).point.id == 10 && graph.node(1).point.id == 20 &&
                graph.node(1).point.values == std::vector<float>{4, 0}, "second point or indices differ");
        require_links(graph, 0, {1});
        require_links(graph, 1, {0});
        require(first.size() == 1 && first.neighbors(0, 0).empty(), "first snapshot was changed");
    });
    run("third insert prunes from the old list owner", [] {
        const auto graph = three_inserts();
        require(graph.size() == 3 && graph.entry_node_index() == 0 &&
                graph.node(0).point.id == 10 && graph.node(1).point.id == 20 &&
                graph.node(2).point.id == 30 && graph.node(2).point.values == std::vector<float>{5, 0},
                "third point or old indices differ");
        require_links(graph, 0, {1});
        require_links(graph, 1, {2});
        require_links(graph, 2, {1});
        std::cout << "Three inserts: 10:[] -> 10<->20 -> 10->20<->30; entry=10\n";
    });
    run("repeated insertion sequence has identical completed data", [] {
        require(same_graph(three_inserts(), three_inserts()), "repeated sequence differs");
    });
    run("directed result matches the exact reference at full width", [] {
        const auto graph = three_inserts();
        std::vector<Point> points;
        for (NodeIndex node_index = 0; node_index < graph.size(); ++node_index) {
            points.push_back(graph.node(node_index).point);
        }
        const std::array<float, 2> q{5, 0};
        const std::vector<Neighbor> expected{{30, 0}, {20, 1}, {10, 25}};
        require(same_results(hnsw_top_k(graph, q, 3, 3), expected) &&
                same_results(vector_search::exact_top_k(points, q, 3), expected),
                "directed full-width query differs from literal exact distances");
    });
    run("disconnecting prune is rejected without changing the input", [] {
        require_rejection<std::invalid_argument>(pair_graph(), {30, {0.5F, 0}}, config);
        std::cout << "Rejected disconnecting insert: original 10<->20 remains usable\n";
    });
    run("new links use nearest distance then source ID", [] {
        const HnswGraph input(2,
            {{{90, {-1, 0}}, 0, {{1}}}, {{10, {1, 0}}, 0, {{0}}}}, 0, {3, 7});
        const auto graph = insert_layer0(input, {50, {0, 0}}, {2, 3, 3});
        require_links(graph, 2, {1, 0});
        require_links(graph, 0, {1, 2});
        require_links(graph, 1, {0, 2});
    });
    run("new list stops at M even when the base cap is larger", [] {
        const HnswGraph input(2,
            {{{90, {-1, 0}}, 0, {{1}}}, {{10, {1, 0}}, 0, {{0}}}}, 0, {3, 7});
        const auto graph = insert_layer0(input, {50, {0, 0}}, {1, 3, 3});
        require_links(graph, 2, {1});
        require_links(graph, 0, {1});
        require_links(graph, 1, {0, 2});
    });
    run("reverse pruning orders equal distances by source ID", [] {
        const HnswGraph input(2,
            {{{90, {0, 0}}, 0, {{1, 2}}}, {{30, {-1, 0}}, 0, {{2}}},
             {{50, {10, 0}}, 0, {{0}}}}, 0, {2, 7});
        const auto graph = insert_layer0(input, {10, {1, 0}}, {1, 3, 4});
        require_links(graph, 0, {3, 1});
        require_links(graph, 1, {2});
        require_links(graph, 2, {0});
        require_links(graph, 3, {0});
    });
    run("owner pruning can keep old neighbors and remove the new reverse link", [] {
        const HnswGraph input(2,
            {{{10, {0, 0}}, 0, {{1, 2}}}, {{20, {0.1F, 0}}, 0, {{0}}},
             {{30, {-0.1F, 0}}, 0, {{0}}}}, 0, {2, 7});
        const auto graph = insert_layer0(input, {40, {10, 0}}, {2, 3, 4});
        // Owner 10 keeps both close old nodes. Node 20 keeps a route to 40.
        require_links(graph, 0, {1, 2});
        require_links(graph, 1, {0, 3});
        require_links(graph, 2, {0});
        require_links(graph, 3, {1, 0});
    });
    run("owner pruning widens extreme coordinates before subtraction", [] {
        const float largest = std::numeric_limits<float>::max();
        const HnswGraph input(2,
            {{{10, {largest, 0}}, 0, {{1, 2}}}, {{20, {-largest, 0}}, 0, {{0}}},
             {{30, {largest, 0}}, 0, {{0}}}}, 0, {2, 7});
        const auto graph = insert_layer0(input, {40, {-largest / 2, 0}}, {2, 3, 4});
        require_links(graph, 0, {2, 3});
        require_links(graph, 1, {0, 3});
        require_links(graph, 2, {0});
        require_links(graph, 3, {1, 0});
    });
    run("positive construction width below a valid M is rejected", [] {
        const HnswGraph graph(2,
            {{{10, {0, 0}}, 0, {{1}}}, {{20, {4, 0}}, 0, {{0}}}}, 0, {2, 7});
        require_rejection<std::invalid_argument>(graph, {30, {5, 0}}, {2, 1, 3});
    });
    run("insertion preserves a nonzero entry and both degree limits", [] {
        const HnswGraph input(2,
            {{{10, {0, 0}}, 0, {{1}}}, {{20, {4, 0}}, 0, {{0}}}}, 1, {2, 7});
        const auto graph = insert_layer0(input, {30, {5, 0}}, {1, 3, 3});
        require(graph.entry_node_index() == 1 && graph.dimension() == 2 &&
                graph.degree_limits().base_max_degree == 2 &&
                graph.degree_limits().upper_max_degree == 7, "input metadata was replaced");
        require_links(graph, 1, {0, 2});
    });
    run("identical vectors with distinct IDs remain distinct nodes", [] {
        const auto first = insert_layer0(empty_graph(), {90, {0, 0}}, config);
        const auto graph = insert_layer0(first, {10, {0, 0}}, config);
        const std::vector<Neighbor> expected{{10, 0}, {90, 0}};
        require(same_results(hnsw_top_k(graph, query, 2, 3), expected),
                "equal vectors lost source identity");
    });
    run("zero and maximum uint64 IDs map to stored positions", [] {
        const auto first = insert_layer0(empty_graph(),
            {std::numeric_limits<std::uint64_t>::max(), {0, 0}}, config);
        const auto graph = insert_layer0(first, {0, {4, 0}}, config);
        require(graph.node(0).point.id == std::numeric_limits<std::uint64_t>::max() &&
                graph.node(1).point.id == 0, "extreme source IDs changed");
        require_links(graph, 0, {1});
        require_links(graph, 1, {0});
    });
    run("large requests do not cause request-sized allocation", [] {
        const auto maximum = std::numeric_limits<std::size_t>::max();
        const auto graph = insert_layer0(pair_graph(), {30, {5, 0}}, {1, maximum, maximum});
        require(graph.size() == 3, "large limits prevented a small valid insert");
        require_links(graph, 1, {2});
    });
    run("result owns new point values copied from an lvalue", [] {
        Point point{10, {0, 0}};
        const auto graph = insert_layer0(empty_graph(), point, config);
        require(point.values == std::vector<float>{0, 0}, "lvalue point was consumed");
        point.values[0] = 999;
        require(graph.node(0).point.values == std::vector<float>{0, 0}, "result aliases caller point");
    });
    run("result survives destruction of the original graph", [] {
        const auto graph = [] {
            const auto input = pair_graph();
            return insert_layer0(input, {30, {5, 0}}, config);
        }();
        require_links(graph, 0, {1});
        require_links(graph, 1, {2});
        require_links(graph, 2, {1});
        require(graph.node(0).point.values == std::vector<float>{0, 0}, "old coordinates were lost");
    });
    run("successful insertion also preserves input storage", [] {
        const auto input = pair_graph();
        const auto before = input;
        const auto old_links = input.neighbors(1, 0);
        const auto* old_point = &input.node(1).point;
        const auto graph = insert_layer0(input, {30, {5, 0}}, config);
        require(graph.size() == 3 && same_graph(input, before) &&
                old_point == &input.node(1).point && old_links.data() == input.neighbors(1, 0).data() &&
                std::ranges::equal(old_links, std::array<NodeIndex, 1>{0}),
                "successful insertion changed the const input or borrowed views");
    });
    run("finite extreme floats are subtracted in double", [] {
        const float largest = std::numeric_limits<float>::max();
        const auto first = insert_layer0(empty_graph(), {10, {largest, 0}}, config);
        const auto graph = insert_layer0(first, {20, {-largest, 0}}, config);
        const std::array<float, 2> q{-largest, 0};
        const double difference = 2.0 * static_cast<double>(largest);
        const auto result = hnsw_top_k(graph, q, 2, 3);
        require(result.size() == 2 && result[1].id == 10 &&
                std::isfinite(result[1].squared_distance) &&
                result[1].squared_distance == difference * difference,
                "float subtraction overflowed before widening");
    });
    run("duplicate source ID is rejected", [] {
        require_rejection<std::invalid_argument>(pair_graph(), {10, {5, 0}}, config);
    });
    const std::array<Point, 5> invalid_points{{
        {30, {}}, {30, {5}},
        {30, {std::numeric_limits<float>::quiet_NaN(), 0}},
        {30, {std::numeric_limits<float>::infinity(), 0}},
        {30, {0, -std::numeric_limits<float>::infinity()}}
    }};
    for (std::size_t point_index = 0; point_index < invalid_points.size(); ++point_index) {
        for (bool empty : {true, false}) {
            const auto name = std::string(empty ? "empty" : "nonempty") +
                " input rejects bad point " + std::to_string(point_index);
            run(name, [&] {
                const auto graph = empty ? empty_graph() : pair_graph();
                require_rejection<std::invalid_argument>(graph, invalid_points[point_index], config);
            });
        }
    }
    const std::array<Layer0InsertConfig, 5> invalid_settings{{
        {0, 3, 3}, {2, 1, 3}, {1, 0, 3}, {1, 3, 0}, {2, 3, 3}
    }};
    for (std::size_t setting_index = 0; setting_index < invalid_settings.size(); ++setting_index) {
        for (bool empty : {true, false}) {
            const auto name = std::string(empty ? "empty" : "nonempty") +
                " input rejects bad settings " + std::to_string(setting_index);
            run(name, [&] {
                const auto graph = empty ? empty_graph() : pair_graph();
                require_rejection<std::invalid_argument>(graph, {30, {5, 0}},
                                                        invalid_settings[setting_index]);
            });
        }
    }
    run("logical capacity rejects a fourth point", [] {
        require_rejection<std::length_error>(three_inserts(), {40, {6, 0}}, config);
    });
    run("logical capacity rejects an input already above the limit", [] {
        require_rejection<std::length_error>(pair_graph(), {30, {5, 0}}, {1, 3, 1});
    });
    run("multi-layer input is rejected", [] {
        const HnswGraph graph(2,
            {{{10, {0, 0}}, 0, {{1}}}, {{20, {4, 0}}, 1, {{0}, {}}}}, 1, {1, 1});
        require_rejection<std::invalid_argument>(graph, {30, {5, 0}}, config);
    });
    std::cout << "HNSW insertion checks: " << cases << " cases, " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
