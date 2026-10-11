#include "vector_search/hnsw_graph.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using vector_search::GraphDegreeLimits;
using vector_search::HnswGraph;
using vector_search::HnswNode;
using vector_search::NodeIndex;

// This fixed graph matches the hand-worked search trace.
std::vector<HnswNode> trace_nodes() {
    return {
        {{10, {9, 0}}, 2, {{1}, {1}, {1}}},
        {{20, {5, 0}}, 2, {{0, 2}, {0, 2}, {0}}},
        {{30, {2, 0}}, 1, {{1, 3, 4}, {1}}},
        {{40, {1, 0}}, 0, {{2, 5}}},
        {{50, {3, 0}}, 0, {{2, 6}}},
        {{60, {4, 0}}, 0, {{3}}},
        {{70, {0, 0}}, 0, {{4}}},
    };
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template<class Exception>
void expect_rejected(auto operation) {
    try {
        operation();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error("invalid input was accepted");
}

bool equal_indices(std::span<const NodeIndex> actual, std::initializer_list<NodeIndex> expected) {
    return std::vector<NodeIndex>(actual.begin(), actual.end()) ==
        std::vector<NodeIndex>(expected);
}

static_assert(std::is_same_v<
    decltype(std::declval<const HnswGraph&>().node(0)), const HnswNode&>);
static_assert(std::is_same_v<
    decltype(std::declval<const HnswGraph&>().neighbors(0, 0)),
    std::span<const NodeIndex>>);

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
    const auto invalid_nodes = [&](std::string_view name, auto change) {
        run(name, [&] {
            auto nodes = trace_nodes();
            change(nodes);
            expect_rejected<std::invalid_argument>([&] {
                (void)HnswGraph(2, std::move(nodes), 0, {3, 2});
            });
        });
    };

    run("manual graph preserves source identities and traced layer links", [] {
        const HnswGraph graph(2, trace_nodes(), 0, {3, 2});
        require(graph.size() == 7 && graph.dimension() == 2 &&
                graph.entry_node_index() == 0, "graph shape or entry changed");
        const std::array<std::uint64_t, 7> source_ids{10, 20, 30, 40, 50, 60, 70};
        std::vector<vector_search::Point> points;
        for (NodeIndex node_index = 0; node_index < graph.size(); ++node_index) {
            require(graph.node(node_index).point.id == source_ids[node_index],
                    "source ID became attached to the wrong node index");
            points.push_back(graph.node(node_index).point);
            for (std::size_t layer = 0; layer <= graph.node(node_index).level; ++layer) {
                for (NodeIndex neighbor_index : graph.neighbors(node_index, layer)) {
                    const auto reverse = graph.neighbors(neighbor_index, layer);
                    bool found = false;
                    for (NodeIndex reverse_index : reverse) {
                        found = found || reverse_index == node_index;
                    }
                    require(found, "the manual fixture lost a reciprocal link");
                }
            }
        }
        require(graph.node(2).level == 1 && equal_indices(graph.neighbors(2, 0), {1, 3, 4}) &&
                equal_indices(graph.neighbors(2, 1), {1}), "node 30 links changed");
        require(equal_indices(graph.neighbors(1, 2), {0}), "top-layer link changed");
        const std::array<float, 2> query{0, 0};
        const auto result = vector_search::exact_top_k(points, query, 2);
        require(result.size() == 2 && result[0].id == 70 && result[0].squared_distance == 0 &&
                result[1].id == 40 && result[1].squared_distance == 1,
                "stored vectors differ from the traced exact reference");
        std::cout << "Fixture: nodes=7, entry_node_index=0, source_id_30_node_index=2\n";
    });
    run("copied construction owns coordinates and neighbor lists", [] {
        auto input = trace_nodes();
        const HnswGraph graph(2, input, 0, {3, 2});
        input[2].point.values[0] = 999;
        input[2].neighbors[0].clear();
        require(graph.node(2).point.values == std::vector<float>{2, 0} &&
                equal_indices(graph.neighbors(2, 0), {1, 3, 4}),
                "caller changes altered owned graph data");
    });
    run("empty graph has no entry", [] {
        const HnswGraph graph(2, {}, std::nullopt, {3, 2});
        require(graph.size() == 0 && !graph.entry_node_index(), "empty graph has an entry");
    });
    run("equal coordinates with distinct source IDs are valid", [] {
        const HnswGraph graph(2,
            {{{8, {1, 0}}, 0, {{1}}}, {{4, {1, 0}}, 0, {{0}}}}, 0, {1, 1});
        require(graph.node(0).point.id == 8 && graph.node(1).point.id == 4,
                "duplicate coordinates merged distinct points");
    });
    run("directed links are valid when every node is reachable", [] {
        const HnswGraph graph(2,
            {{{8, {0, 0}}, 0, {{1}}}, {{4, {1, 0}}, 0, {{2}}},
             {{3, {2, 0}}, 0, {{}}}}, 0, {1, 1});
        require(equal_indices(graph.neighbors(0, 0), {1}) &&
                graph.neighbors(2, 0).empty(), "directed graph changed");
    });
    run("single node can have empty lists in upper layers", [] {
        const HnswGraph graph(2,
            {{{std::numeric_limits<std::uint64_t>::max(), {0, 0}}, 2, {{}, {}, {}}}},
            0, {1, 1});
        require(graph.node(0).point.id == std::numeric_limits<std::uint64_t>::max() &&
                graph.neighbors(0, 2).empty(), "source ID or empty upper list changed");
    });
    run("another highest-level node can be the entry", [] {
        const HnswGraph graph(2, trace_nodes(), 1, {3, 2});
        require(graph.entry_node_index() == 1, "valid alternate entry changed");
    });
    run("zero dimension is rejected", [] {
        expect_rejected<std::invalid_argument>([] {
            (void)HnswGraph(0, {}, std::nullopt, {3, 2});
        });
    });
    run("zero base degree limit is rejected", [] {
        expect_rejected<std::invalid_argument>([] {
            (void)HnswGraph(2, trace_nodes(), 0, {0, 2});
        });
    });
    run("zero upper degree limit is rejected", [] {
        expect_rejected<std::invalid_argument>([] {
            (void)HnswGraph(2, trace_nodes(), 0, {3, 0});
        });
    });
    invalid_nodes("duplicate source ID is rejected", [](auto& nodes) {
        nodes[1].point.id = 10;
    });
    invalid_nodes("wrong coordinate count is rejected", [](auto& nodes) {
        nodes[1].point.values.pop_back();
    });
    invalid_nodes("NaN coordinate is rejected", [](auto& nodes) {
        nodes[1].point.values[0] = std::numeric_limits<float>::quiet_NaN();
    });
    invalid_nodes("infinite coordinate is rejected", [](auto& nodes) {
        nodes[1].point.values[0] = std::numeric_limits<float>::infinity();
    });
    invalid_nodes("missing layer zero is rejected", [](auto& nodes) {
        nodes[3].neighbors.clear();
    });
    invalid_nodes("level and list count mismatch is rejected", [](auto& nodes) {
        nodes[2].neighbors.push_back({});
    });
    invalid_nodes("maximum size_t level cannot wrap the shape check", [](auto& nodes) {
        nodes[3].level = std::numeric_limits<std::size_t>::max();
    });
    run("missing entry is rejected for a nonempty graph", [] {
        expect_rejected<std::invalid_argument>([] {
            (void)HnswGraph(2, trace_nodes(), std::nullopt, {3, 2});
        });
    });
    run("entry is rejected for an empty graph", [] {
        expect_rejected<std::invalid_argument>([] {
            (void)HnswGraph(2, {}, 0, {3, 2});
        });
    });
    run("out-of-range entry is rejected", [] {
        expect_rejected<std::invalid_argument>([] {
            (void)HnswGraph(2, trace_nodes(), 7, {3, 2});
        });
    });
    run("entry below the highest level is rejected", [] {
        expect_rejected<std::invalid_argument>([] {
            (void)HnswGraph(2, trace_nodes(), 2, {3, 2});
        });
    });
    invalid_nodes("out-of-range neighbor is rejected", [](auto& nodes) {
        nodes[3].neighbors[0] = {2, 7};
    });
    invalid_nodes("self-link is rejected", [](auto& nodes) {
        nodes[3].neighbors[0].push_back(3);
    });
    invalid_nodes("duplicate link is rejected", [](auto& nodes) {
        nodes[3].neighbors[0].push_back(2);
    });
    invalid_nodes("target absent from an upper layer is rejected", [](auto& nodes) {
        nodes[2].neighbors[1].push_back(3);
    });
    run("base degree above its limit is rejected", [] {
        expect_rejected<std::invalid_argument>([] {
            (void)HnswGraph(2, trace_nodes(), 0, {2, 2});
        });
    });
    run("upper degree above its limit is rejected", [] {
        expect_rejected<std::invalid_argument>([] {
            (void)HnswGraph(2, trace_nodes(), 0, {3, 1});
        });
    });
    invalid_nodes("node unreachable from the entry is rejected", [](auto& nodes) {
        // Node 60 can reach the entry, but the entry cannot reach node 60.
        nodes[3].neighbors[0] = {2};
    });
    run("invalid node read throws out_of_range", [] {
        const HnswGraph graph(2, trace_nodes(), 0, {3, 2});
        expect_rejected<std::out_of_range>([&] { (void)graph.node(7); });
    });
    run("invalid neighbor-node read throws out_of_range", [] {
        const HnswGraph graph(2, trace_nodes(), 0, {3, 2});
        expect_rejected<std::out_of_range>([&] {
            (void)graph.neighbors(std::numeric_limits<NodeIndex>::max(), 0);
        });
    });
    run("absent layer read throws out_of_range", [] {
        const HnswGraph graph(2, trace_nodes(), 0, {3, 2});
        expect_rejected<std::out_of_range>([&] { (void)graph.neighbors(3, 1); });
    });
    run("move construction leaves a readable empty source", [] {
        HnswGraph source(2, trace_nodes(), 0, {3, 2});
        const HnswGraph graph(std::move(source));
        require(graph.size() == 7 && graph.entry_node_index() == 0 &&
                source.size() == 0 && !source.entry_node_index(),
                "move lost data or left a stale source entry");
    });
    run("assignment transfers a valid graph and empties the moved source", [] {
        HnswGraph source(2, trace_nodes(), 0, {3, 2});
        HnswGraph graph(3, {}, std::nullopt, {1, 1});
        graph = source;
        source = std::move(graph);
        require(source.dimension() == 2 && source.size() == 7 &&
                equal_indices(source.neighbors(2, 0), {1, 3, 4}) &&
                graph.size() == 0 && !graph.entry_node_index(),
                "assignment lost graph ownership or left a stale entry");
    });
    std::cout << "HNSW graph checks: " << cases << " cases, " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
