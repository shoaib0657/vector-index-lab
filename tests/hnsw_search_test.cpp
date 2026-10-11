#include "vector_search/hnsw_search.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using vector_search::HnswGraph;
using vector_search::HnswNode;
using vector_search::HnswSearchTrace;
using vector_search::Neighbor;
using vector_search::NodeIndex;
using vector_search::hnsw_top_k;

const std::array<float, 2> query{0, 0};

// Keep this literal fixture independent of search implementation details.
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

HnswGraph trace_graph() {
    return HnswGraph(2, trace_nodes(), 0, {3, 2});
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool same_neighbors(std::span<const Neighbor> left, std::span<const Neighbor> right) {
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

void require_neighbors(
    std::span<const Neighbor> actual, std::initializer_list<Neighbor> expected) {
    require(same_neighbors(actual, std::span<const Neighbor>(expected.begin(), expected.size())),
            "neighbor IDs, distances, or order differ");
}

void require_ids(
    const std::vector<std::uint64_t>& actual, std::initializer_list<std::uint64_t> expected) {
    require(actual == std::vector<std::uint64_t>(expected), "source ID sequence differs");
}

bool same_trace(const HnswSearchTrace& left, const HnswSearchTrace& right) {
    if (left.layers.size() != right.layers.size()) {
        return false;
    }
    for (std::size_t layer_index = 0; layer_index < left.layers.size(); ++layer_index) {
        const auto& a = left.layers[layer_index];
        const auto& b = right.layers[layer_index];
        if (a.layer != b.layer || a.entry_source_id != b.entry_source_id ||
            a.visited_source_ids != b.visited_source_ids ||
            a.expanded_source_ids != b.expanded_source_ids ||
            !same_neighbors(a.retained_neighbors, b.retained_neighbors) ||
            a.stopped_candidate.has_value() != b.stopped_candidate.has_value()) {
            return false;
        }
        if (a.stopped_candidate &&
            (a.stopped_candidate->id != b.stopped_candidate->id ||
             a.stopped_candidate->squared_distance != b.stopped_candidate->squared_distance)) {
            return false;
        }
    }
    return true;
}

HnswSearchTrace sentinel_trace() {
    return {{{99, 999, {999, 888}, {999}, {{888, 7}}, Neighbor{999, 8}}}};
}

void expect_invalid(auto operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("invalid query parameters were accepted");
}

HnswGraph equal_distance_graph() {
    return HnswGraph(2,
        {{{90, {2, 0}}, 0, {{1}}}, {{10, {-2, 0}}, 0, {{}}}}, 0, {1, 1});
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

    run("traced query matches both upper layers and top two", [] {
        const auto graph = trace_graph();
        HnswSearchTrace trace;
        const auto result = hnsw_top_k(graph, query, 2, 3, &trace);
        require_neighbors(result, {{70, 0}, {40, 1}});
        require(trace.layers.size() == 3, "trace does not contain three layer calls");
        const auto& top = trace.layers[0];
        require(top.layer == 2 && top.entry_source_id == 10, "top entry differs");
        require_ids(top.visited_source_ids, {10, 20});
        require_ids(top.expanded_source_ids, {10, 20});
        require_neighbors(top.retained_neighbors, {{20, 25}});
        require(!top.stopped_candidate, "top layer has an unexpected stop candidate");
        const auto& middle = trace.layers[1];
        require(middle.layer == 1 && middle.entry_source_id == 20, "middle entry differs");
        require_ids(middle.visited_source_ids, {20, 10, 30});
        require_ids(middle.expanded_source_ids, {20, 30});
        require_neighbors(middle.retained_neighbors, {{30, 4}});
        require(!middle.stopped_candidate, "middle layer has an unexpected stop candidate");
        require(trace.layers[2].layer == 0 && trace.layers[2].entry_source_id == 30,
                "upper descent did not reach source ID 30");
        std::cout << "Traced query result: 70:0 40:1\n";
    });
    run("bottom discovery includes rejected node sixty", [] {
        HnswSearchTrace trace;
        (void)hnsw_top_k(trace_graph(), query, 2, 3, &trace);
        require_ids(trace.layers.at(2).visited_source_ids, {30, 20, 40, 50, 60, 70});
    });
    run("bottom expansion excludes rejected and stopped candidates", [] {
        HnswSearchTrace trace;
        (void)hnsw_top_k(trace_graph(), query, 2, 3, &trace);
        require_ids(trace.layers.at(2).expanded_source_ids, {30, 40, 50, 70});
    });
    run("best set and stale candidate reproduce the stopping comparison", [] {
        HnswSearchTrace trace;
        (void)hnsw_top_k(trace_graph(), query, 2, 3, &trace);
        const auto& bottom = trace.layers.at(2);
        require_neighbors(bottom.retained_neighbors, {{70, 0}, {40, 1}, {30, 4}});
        require(bottom.stopped_candidate && bottom.stopped_candidate->id == 20 &&
                bottom.stopped_candidate->squared_distance == 25, "stale candidate differs");
        require(bottom.stopped_candidate->squared_distance >
                bottom.retained_neighbors.back().squared_distance, "stop threshold differs");
    });
    run("width one misses the route through node fifty", [] {
        require_neighbors(hnsw_top_k(trace_graph(), query, 1, 1), {{40, 1}});
    });
    run("width three finds node seventy", [] {
        require_neighbors(hnsw_top_k(trace_graph(), query, 1, 3), {{70, 0}});
    });
    run("wide search agrees with the exact baseline", [] {
        const auto nodes = trace_nodes();
        std::vector<vector_search::Point> points;
        for (const auto& node : nodes) {
            points.push_back(node.point);
        }
        const auto exact = vector_search::exact_top_k(points, query, 7);
        require(same_neighbors(hnsw_top_k(trace_graph(), query, 7, 7), exact),
                "wide search differs from exact search");
    });
    run("k beyond graph size returns all reachable nodes", [] {
        require_neighbors(hnsw_top_k(trace_graph(), query, 10, 10),
            {{70, 0}, {40, 1}, {30, 4}, {50, 9}, {60, 16}, {20, 25}, {10, 81}});
    });
    run("maximum requested k and ef do not drive allocation", [] {
        const auto maximum = std::numeric_limits<std::size_t>::max();
        require_neighbors(hnsw_top_k(trace_graph(), query, maximum, maximum),
            {{70, 0}, {40, 1}, {30, 4}, {50, 9}, {60, 16}, {20, 25}, {10, 81}});
    });
    run("repeated and different queries use independent scratch state", [] {
        const auto graph = trace_graph();
        HnswSearchTrace first;
        HnswSearchTrace second;
        const auto initial = hnsw_top_k(graph, query, 2, 3, &first);
        const std::array<float, 2> another_query{9, 0};
        require_neighbors(hnsw_top_k(graph, another_query, 1, 3), {{10, 0}});
        const auto repeated = hnsw_top_k(graph, query, 2, 3, &second);
        require_neighbors(initial, {{70, 0}, {40, 1}});
        require(same_neighbors(initial, repeated) && same_trace(first, second),
                "query state leaked into a later call");
    });
    run("tracing does not change returned neighbors", [] {
        const auto graph = trace_graph();
        HnswSearchTrace trace;
        const auto traced = hnsw_top_k(graph, query, 2, 3, &trace);
        require_neighbors(traced, {{70, 0}, {40, 1}});
        require(same_neighbors(traced, hnsw_top_k(graph, query, 2, 3)),
                "trace changed the search result");
    });
    run("a successful query replaces a prior trace", [] {
        auto trace = sentinel_trace();
        (void)hnsw_top_k(trace_graph(), query, 2, 3, &trace);
        require(trace.layers.size() == 3 && trace.layers.front().entry_source_id == 10,
                "old trace layers were retained");
    });
    run("an invalid query preserves the complete prior trace", [] {
        auto trace = sentinel_trace();
        const auto original = trace;
        const std::array<float, 1> bad_query{0};
        expect_invalid([&] { (void)hnsw_top_k(trace_graph(), bad_query, 1, 1, &trace); });
        require(same_trace(trace, original), "throwing query changed its trace");
    });
    run("zero k skips bad query and ef and clears trace", [] {
        auto trace = sentinel_trace();
        const std::array<float, 1> bad_query{std::numeric_limits<float>::quiet_NaN()};
        require(hnsw_top_k(trace_graph(), bad_query, 0, 0, &trace).empty(),
                "zero k returned neighbors");
        require(trace.layers.empty(), "empty result kept a prior trace");
    });
    run("empty graph skips bad query and ef and clears trace", [] {
        const HnswGraph graph(2, {}, std::nullopt, {1, 1});
        auto trace = sentinel_trace();
        const std::array<float, 1> bad_query{std::numeric_limits<float>::infinity()};
        require(hnsw_top_k(graph, bad_query, 3, 0, &trace).empty(),
                "empty graph returned neighbors");
        require(trace.layers.empty(), "empty graph kept a prior trace");
    });
    run("moved-from graph follows the empty graph convention", [] {
        auto source = trace_graph();
        const auto destination = std::move(source);
        auto trace = sentinel_trace();
        require(hnsw_top_k(source, {}, 1, 0, &trace).empty() && trace.layers.empty(),
                "moved source retained query state");
        require_neighbors(hnsw_top_k(destination, query, 2, 3), {{70, 0}, {40, 1}});
    });
    run("single node supports the largest source ID", [] {
        const auto source_id = std::numeric_limits<std::uint64_t>::max();
        const HnswGraph graph(2, {{{source_id, {0, 0}}, 0, {{}}}}, 0, {1, 1});
        require_neighbors(hnsw_top_k(graph, query, 1, 1), {{source_id, 0}});
    });
    run("empty upper lists descend without unsigned underflow", [] {
        const HnswGraph graph(2, {{{17, {0, 0}}, 2, {{}, {}, {}}}}, 0, {1, 1});
        HnswSearchTrace trace;
        require_neighbors(hnsw_top_k(graph, query, 1, 1, &trace), {{17, 0}});
        require(trace.layers.size() == 3, "empty upper lists lost layer calls");
        for (std::size_t layer_index = 0; layer_index < 3; ++layer_index) {
            require(trace.layers[layer_index].layer == 2 - layer_index &&
                    trace.layers[layer_index].entry_source_id == 17, "layer countdown differs");
        }
    });
    run("directed level-zero links use indices rather than large IDs", [] {
        const auto source_id = std::numeric_limits<std::uint64_t>::max() - 1;
        const HnswGraph graph(2,
            {{{source_id, {9, 0}}, 0, {{1}}}, {{5, {0, 0}}, 0, {{}}}}, 0, {1, 1});
        HnswSearchTrace trace;
        require_neighbors(hnsw_top_k(graph, query, 1, 1, &trace), {{5, 0}});
        require(trace.layers.size() == 1, "level zero performed upper descent");
        require_ids(trace.layers[0].visited_source_ids, {source_id, 5});
    });
    run("wrong query dimension is rejected", [] {
        const std::array<float, 1> bad_query{0};
        expect_invalid([&] { (void)hnsw_top_k(trace_graph(), bad_query, 1, 1); });
    });
    run("empty query is rejected for active search", [] {
        expect_invalid([] { (void)hnsw_top_k(trace_graph(), {}, 1, 1); });
    });
    run("NaN query coordinate is rejected", [] {
        const std::array<float, 2> bad_query{0, std::numeric_limits<float>::quiet_NaN()};
        expect_invalid([&] { (void)hnsw_top_k(trace_graph(), bad_query, 1, 1); });
    });
    run("positive infinity query coordinate is rejected", [] {
        const std::array<float, 2> bad_query{std::numeric_limits<float>::infinity(), 0};
        expect_invalid([&] { (void)hnsw_top_k(trace_graph(), bad_query, 1, 1); });
    });
    run("negative infinity query coordinate is rejected", [] {
        const std::array<float, 2> bad_query{0, -std::numeric_limits<float>::infinity()};
        expect_invalid([&] { (void)hnsw_top_k(trace_graph(), bad_query, 1, 1); });
    });
    run("zero ef is rejected for positive k", [] {
        expect_invalid([] { (void)hnsw_top_k(trace_graph(), query, 1, 0); });
    });
    run("ef below requested k is rejected", [] {
        expect_invalid([] { (void)hnsw_top_k(trace_graph(), query, 2, 1); });
    });
    run("requested k still sets the ef rule beyond graph size", [] {
        expect_invalid([] { (void)hnsw_top_k(trace_graph(), query, 10, 7); });
    });
    run("full best set rejects an equal-distance smaller ID", [] {
        HnswSearchTrace trace;
        require_neighbors(hnsw_top_k(equal_distance_graph(), query, 1, 1, &trace), {{90, 4}});
        require_ids(trace.layers.at(0).visited_source_ids, {90, 10});
        require_ids(trace.layers.at(0).expanded_source_ids, {90});
    });
    run("space admits equal distances and results use ID order", [] {
        const auto graph = equal_distance_graph();
        require_neighbors(hnsw_top_k(graph, query, 2, 2), {{10, 4}, {90, 4}});
        require_neighbors(hnsw_top_k(graph, query, 1, 2), {{10, 4}});
    });
    run("equal stop distance expands and heap ties remove the larger ID", [] {
        // A tied pending node leads to the exact match.
        const HnswGraph graph(2,
            {{{80, {3, 0}}, 0, {{1, 2}}}, {{90, {-2, 0}}, 0, {{}}},
             {{10, {2, 0}}, 0, {{3}}}, {{5, {0, 0}}, 0, {{}}}}, 0, {2, 1});
        HnswSearchTrace trace;
        require_neighbors(hnsw_top_k(graph, query, 2, 2, &trace), {{5, 0}, {10, 4}});
        require_ids(trace.layers.at(0).expanded_source_ids, {80, 10, 5, 90});
        require(!trace.layers[0].stopped_candidate, "equal distance caused an early stop");
    });
    run("query leaves stored vectors links levels and entry unchanged", [] {
        const auto before = trace_nodes();
        const HnswGraph graph(2, before, 0, {3, 2});
        HnswSearchTrace trace;
        require_neighbors(hnsw_top_k(graph, query, 2, 3, &trace), {{70, 0}, {40, 1}});
        require(graph.size() == before.size() && graph.dimension() == 2 &&
                graph.entry_node_index() == 0, "graph metadata changed");
        for (NodeIndex node_index = 0; node_index < graph.size(); ++node_index) {
            const auto& actual = graph.node(node_index);
            require(actual.point.id == before[node_index].point.id &&
                    actual.point.values == before[node_index].point.values &&
                    actual.level == before[node_index].level &&
                    actual.neighbors == before[node_index].neighbors, "stored node changed");
        }
    });
    run("trace owns its values after the graph is destroyed", [] {
        const auto trace = [] {
            HnswSearchTrace local_trace;
            const auto local_graph = trace_graph();
            (void)hnsw_top_k(local_graph, query, 2, 3, &local_trace);
            return local_trace;
        }();
        const auto copied = trace;
        require_ids(copied.layers.at(2).visited_source_ids, {30, 20, 40, 50, 60, 70});
        require_neighbors(copied.layers[2].retained_neighbors, {{70, 0}, {40, 1}, {30, 4}});
    });
    run("fractional coordinates accumulate both dimensions", [] {
        const HnswGraph graph(2, {{{8, {1.5F, -1.5F}}, 0, {{}}}}, 0, {1, 1});
        const std::array<float, 2> fractional_query{0.5F, 0.5F};
        require_neighbors(hnsw_top_k(graph, fractional_query, 1, 1), {{8, 5}});
    });
    run("float coordinates are widened before subtraction", [] {
        const float maximum = std::numeric_limits<float>::max();
        const HnswGraph graph(1, {{{8, {maximum}}, 0, {{}}}}, 0, {1, 1});
        const std::array<float, 1> distant_query{-maximum};
        const auto result = hnsw_top_k(graph, distant_query, 1, 1);
        require(result.size() == 1 && result[0].id == 8 &&
                std::isfinite(result[0].squared_distance) &&
                result[0].squared_distance > static_cast<double>(maximum),
                "float subtraction overflowed before distance accumulation");
    });
    run("invalid ef preserves a prior trace", [] {
        auto trace = sentinel_trace();
        const auto original = trace;
        expect_invalid([&] { (void)hnsw_top_k(trace_graph(), query, 2, 1, &trace); });
        require(same_trace(trace, original), "invalid ef changed its trace");
    });

    run("full width retains reachability after directed upper descent", [] {
        const HnswGraph graph(1,
            {{{10, {9}}, 1, {{1}, {1}}}, {{20, {0}}, 1, {{}, {}}}}, 0, {1, 1});
        const std::array<float, 1> directed_query{0};
        HnswSearchTrace trace;
        require_neighbors(hnsw_top_k(graph, directed_query, 2, 2, &trace), {{20, 0}, {10, 81}});
        require(trace.layers.size() == 2 && trace.layers[0].entry_source_id == 10 &&
                trace.layers[1].entry_source_id == 10, "full width lost the stored entry");
        require_neighbors(hnsw_top_k(graph, directed_query, 3, 3), {{20, 0}, {10, 81}});
        const auto maximum = std::numeric_limits<std::size_t>::max();
        require_neighbors(hnsw_top_k(graph, directed_query, maximum, maximum),
                          {{20, 0}, {10, 81}});
    });
    run("narrow search retains directed upper descent", [] {
        const HnswGraph graph(1,
            {{{10, {9}}, 1, {{1}, {1}}}, {{20, {0}}, 1, {{}, {}}}}, 0, {1, 1});
        const std::array<float, 1> directed_query{0};
        HnswSearchTrace trace;
        require_neighbors(hnsw_top_k(graph, directed_query, 1, 1, &trace), {{20, 0}});
        require(trace.layers.size() == 2 && trace.layers[1].entry_source_id == 20,
                "narrow search discarded the descended entry");
    });

    std::cout << "HNSW search checks: " << cases << " cases, " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
