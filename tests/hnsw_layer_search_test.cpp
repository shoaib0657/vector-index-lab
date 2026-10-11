#include "hnsw_search_internal.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
using namespace vector_search;
using detail::search_layer;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Node 30 supplies the only base route to 40 in this directed graph.
HnswGraph route_graph() {
    return HnswGraph(2,
        {{{10, {2, 0}}, 1, {{}, {}}}, {{20, {3, 0}}, 1, {{}, {}}},
         {{30, {10, 0}}, 2, {{0, 1, 3}, {0, 1}, {}}},
         {{40, {0, 0}}, 0, {{}}}}, 2, {3, 2});
}

void require_result(std::span<const detail::ScoredNode> result,
                    std::initializer_list<NodeIndex> indices,
                    std::initializer_list<double> distances) {
    require(result.size() == indices.size() && result.size() == distances.size(),
            "retained count differs");
    auto index = indices.begin();
    auto distance = distances.begin();
    for (const auto& node : result) {
        require(node.node_index == *index++ && node.squared_distance == *distance++,
                "retained indices or literal distances differ");
    }
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
    const auto graph = route_graph();
    const std::array<float, 2> query{1, 0};
    run("upper singleton returns the full W in distance and source-ID order", [&] {
        const std::array<NodeIndex, 1> entries{2};
        LayerSearchTrace trace{};
        const auto result = search_layer(graph, query, entries, 1, 3, &trace);
        require_result(result, {0, 1, 2}, {1, 4, 81});
        require(trace.entry_source_id == 30 && trace.layer == 1 &&
                trace.expanded_source_ids == std::vector<std::uint64_t>{30, 10, 20},
                "singleton trace differs");
    });
    run("every retained entry can supply a base route", [&] {
        const std::array<NodeIndex, 3> entries{0, 1, 2};
        const auto result = search_layer(graph, query, entries, 0, 3);
        require_result(result, {0, 3, 1}, {1, 1, 4});
        require(result[0].source_id == 10 && result[1].source_id == 40,
                "equal distances lost source-ID order");
    });
    run("selected-only and singleton entries do not silently use stored entry", [&] {
        const std::array<NodeIndex, 2> selected{0, 1};
        const std::array<NodeIndex, 1> nearest{0};
        require_result(search_layer(graph, query, selected, 0, 3), {0, 1}, {1, 4});
        require_result(search_layer(graph, query, nearest, 0, 4), {0}, {1});
    });
    run("each call starts with fresh visited and retained state", [&] {
        const std::array<NodeIndex, 3> entries{2, 1, 0};
        (void)search_layer(graph, std::array<float, 2>{10, 0}, entries, 0, 3);
        require_result(search_layer(graph, query, entries, 0, 3), {0, 3, 1}, {1, 1, 4});
    });
    run("a successful singleton trace replaces old trace data", [&] {
        const std::array<NodeIndex, 1> entries{0};
        LayerSearchTrace trace{};
        trace.visited_source_ids = {999};
        trace.expanded_source_ids = {999};
        trace.retained_neighbors = {{999, 9}};
        trace.stopped_candidate = Neighbor{999, 9};
        (void)search_layer(graph, query, entries, 0, 1, &trace);
        require(trace.visited_source_ids == std::vector<std::uint64_t>{10} &&
                trace.expanded_source_ids == std::vector<std::uint64_t>{10} &&
                trace.retained_neighbors.size() == 1 &&
                trace.retained_neighbors[0].id == 10 && !trace.stopped_candidate,
                "trace accumulated an earlier call");
    });

    struct BadInput {
        std::vector<NodeIndex> entries;
        std::size_t layer;
        std::size_t ef;
        std::vector<float> query;
    };
    const std::array<BadInput, 11> bad{{
        {{}, 0, 3, {1, 0}}, {{0}, 0, 0, {1, 0}}, {{0, 1}, 0, 1, {1, 0}},
        {{0, 0}, 0, 3, {1, 0}}, {{4}, 0, 3, {1, 0}},
        {{std::numeric_limits<NodeIndex>::max()}, 0, 3, {1, 0}},
        {{3}, 1, 3, {1, 0}}, {{2}, 3, 3, {1, 0}},
        {{0}, 0, 3, {1}}, {{0}, 0, 3, {std::numeric_limits<float>::quiet_NaN(), 0}},
        {{0}, 0, 3, {0, std::numeric_limits<float>::infinity()}}
    }};
    for (std::size_t case_index = 0; case_index < bad.size(); ++case_index) {
        run("bad entries or query " + std::to_string(case_index), [&] {
            const auto& input = bad[case_index];
            LayerSearchTrace trace{};
            trace.layer = 9;
            trace.entry_source_id = 999;
            trace.visited_source_ids = {999};
            bool rejected = false;
            try {
                (void)search_layer(graph, input.query, input.entries, input.layer, input.ef, &trace);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            require(rejected, "invalid internal search input was accepted");
            require(trace.layer == 9 && trace.entry_source_id == 999 &&
                    trace.visited_source_ids == std::vector<std::uint64_t>{999},
                    "failed search changed the caller trace");
        });
    }
    run("duplicate entries without a trace are rejected", [&] {
        const std::array<NodeIndex, 2> entries{0, 0};
        bool rejected = false;
        try {
            (void)search_layer(graph, query, entries, 0, 3);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "duplicate entries were accepted without a trace");
    });
    run("multiple entries reject the singular trace schema without changing it", [&] {
        const std::array<NodeIndex, 2> entries{0, 1};
        LayerSearchTrace trace{};
        trace.entry_source_id = 999;
        bool rejected = false;
        try {
            (void)search_layer(graph, query, entries, 0, 3, &trace);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected && trace.entry_source_id == 999, "singular trace was misleading");
    });
    std::cout << "HNSW layer-search checks: " << cases << " cases, " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
