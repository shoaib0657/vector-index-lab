#include "vector_search/hnsw_insert.h"
#include "vector_search/hnsw_search.h"
#include "hnsw_insert_internal.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace {
using namespace vector_search;
using detail::insert_at_level;
using detail::level_from_word;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <class Exception, class Action>
void require_throws(Action action) {
    bool rejected = false;
    try {
        action();
    } catch (const Exception&) {
        rejected = true;
    }
    require(rejected, "required exception was not thrown");
}

bool same_config(HnswBuildConfig a, HnswBuildConfig b) {
    return a.neighbors_per_node == b.neighbors_per_node &&
        a.ef_construction == b.ef_construction && a.max_nodes == b.max_nodes &&
        a.level_cap == b.level_cap && a.seed == b.seed;
}

bool same_graph(const HnswGraph& a, const HnswGraph& b) {
    if (a.size() != b.size() || a.dimension() != b.dimension() ||
        a.entry_node_index() != b.entry_node_index() ||
        a.degree_limits().base_max_degree != b.degree_limits().base_max_degree ||
        a.degree_limits().upper_max_degree != b.degree_limits().upper_max_degree) {
        return false;
    }
    for (NodeIndex index = 0; index < a.size(); ++index) {
        const auto& left = a.node(index);
        const auto& right = b.node(index);
        if (left.point.id != right.point.id || left.point.values != right.point.values ||
            left.level != right.level || left.neighbors != right.neighbors) {
            return false;
        }
    }
    return true;
}

void require_links(const HnswGraph& graph, NodeIndex node_index,
                   std::initializer_list<std::vector<NodeIndex>> lists) {
    require(graph.node(node_index).neighbors == std::vector<std::vector<NodeIndex>>(lists),
            "literal layer lists differ");
}

HnswGraph route_graph() {
    return HnswGraph(2,
        {{{10, {2, 0}}, 1, {{}, {}}}, {{20, {3, 0}}, 1, {{}, {}}},
         {{30, {10, 0}}, 2, {{0, 1, 3}, {0, 1}, {}}},
         {{40, {0, 0}}, 0, {{}}}}, 2, {3, 2});
}

HnswBuildState first_three(GraphDegreeLimits limits = {4, 2}) {
    HnswBuildState state(2, limits, {2, 3, 8, 4, 42});
    state = state.insert({10, {0, 0}});
    state = state.insert({20, {4, 0}});
    return state.insert({30, {5, 0}});
}

// The fourth seed-42 draw is level 2. A failed attempt must not consume it.
void require_preserved_after_failure(const HnswBuildState& state, Point point) {
    const auto before = state.graph();
    const auto settings = state.config();
    const auto* old_node = &state.graph().node(0);
    const auto old_links = state.graph().neighbors(0, 0);
    const std::array<float, 2> query{0, 0};
    const auto old_result = hnsw_top_k(state.graph(), query, 3, 3);
    require_throws<std::invalid_argument>([&] { (void)state.insert(std::move(point)); });
    require(same_graph(state.graph(), before) && same_config(state.config(), settings) &&
            old_node == &state.graph().node(0) &&
            old_links.data() == state.graph().neighbors(0, 0).data() &&
            std::ranges::equal(old_links, before.neighbors(0, 0)),
            "failed insertion changed stored data or borrowed views");
    const auto result = hnsw_top_k(state.graph(), query, 3, 3);
    require(result.size() == old_result.size(), "failed insertion changed query count");
    for (std::size_t index = 0; index < result.size(); ++index) {
        require(result[index].id == old_result[index].id &&
                result[index].squared_distance == old_result[index].squared_distance,
                "failed insertion changed query results");
    }
    const auto continued = state.insert({40, {9, 0}});
    require(continued.graph().node(3).level == 2 && continued.graph().entry_node_index() == 3 &&
            same_graph(continued.graph(), first_three(state.graph().degree_limits()).insert({40, {9, 0}}).graph()),
            "failed insertion advanced the generator");
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
    run("word endpoints clip before integer conversion", [] {
        require(level_from_word(0, 2, 4) == 4 && level_from_word(0, 2, 32) == 32 &&
                level_from_word(std::numeric_limits<std::uint64_t>::max(), 2, 4) == 0 &&
                level_from_word(0, 2, 0) == 0, "level endpoint differs");
    });
    run("literal random words map to the reviewed level sequence", [] {
        const std::array<std::uint64_t, 8> words{
            13930160852258120406ULL, 11788048577503494824ULL, 13874630024467741450ULL,
            2513787319205155662ULL, 16662371453428439381ULL, 1735254072534978428ULL,
            10598951352238613536ULL, 6878563960102566144ULL};
        const std::array<std::size_t, 8> levels{0, 0, 0, 2, 0, 3, 0, 1};
        for (std::size_t index = 0; index < words.size(); ++index) {
            require(level_from_word(words[index], 2, 4) == levels[index], "literal sampled level differs");
        }
    });
    run("invalid sampler settings are rejected", [] {
        require_throws<std::invalid_argument>([] { (void)level_from_word(0, 1, 4); });
        require_throws<std::invalid_argument>([] { (void)level_from_word(0, 2, 33); });
    });
    run("multi-layer insertion carries unselected retained entries down", [] {
        const auto input = route_graph();
        const auto before = input;
        const auto graph = insert_at_level(input, {50, {1, 0}}, {2, 3, 5}, 1);
        require(graph.size() == 5 && graph.entry_node_index() == 2 && same_graph(input, before),
                "old snapshot or entry changed");
        require_links(graph, 0, {{4}, {4}});
        require_links(graph, 1, {{}, {4}});
        require_links(graph, 2, {{0, 1, 3}, {0, 1}, {}});
        require_links(graph, 3, {{4}});
        require_links(graph, 4, {{0, 3}, {0, 1}});
    });
    run("paper fixture prunes independently from upper and base owners", [] {
        const HnswGraph input(2,
            {{{10, {9, 0}}, 2, {{1}, {1}, {1}}},
             {{20, {5, 0}}, 2, {{0, 2}, {0, 2}, {0}}},
             {{30, {2, 0}}, 1, {{1, 3, 4}, {1}}},
             {{40, {1, 0}}, 0, {{2, 5}}}, {{50, {3, 0}}, 0, {{2, 6}}},
             {{60, {4, 0}}, 0, {{3}}}, {{70, {0, 0}}, 0, {{4}}}}, 0, {3, 2});
        const auto graph = insert_at_level(input, {80, {1.25F, 0}}, {2, 3, 8}, 1);
        require(graph.entry_node_index() == 0, "lower insertion promoted the entry");
        require_links(graph, 0, {{1}, {1}, {1}});
        require_links(graph, 1, {{0, 2}, {2, 7}, {0}});
        require_links(graph, 2, {{7, 3, 4}, {1, 7}});
        require_links(graph, 3, {{2, 5, 7}});
        require_links(graph, 4, {{2, 6}});
        require_links(graph, 5, {{3}});
        require_links(graph, 6, {{4}});
        require_links(graph, 7, {{3, 2}, {2, 1}});
    });
    for (std::size_t level : {0U, 2U, 3U}) {
        run("controlled level and entry " + std::to_string(level), [=] {
            // Base routes stay connected when the new node becomes the entry.
            const HnswGraph input(2,
                {{{10, {2, 0}}, 2, {{1, 2}, {}, {}}},
                 {{20, {3, 0}}, 2, {{0, 2}, {}, {}}},
                 {{30, {10, 0}}, 2, {{0, 1}, {0, 1}, {0, 1}}}}, 2, {3, 2});
            const auto graph = insert_at_level(input, {50, {1, 0}}, {2, 3, 4}, level);
            require(graph.node(3).level == level && graph.node(3).neighbors.size() == level + 1 &&
                    graph.entry_node_index() == (level > 2 ? 3 : 2), "entry promotion differs");
            if (level > 2) {
                require(graph.neighbors(3, 3).empty(), "new top layer links an absent old node");
            }
        });
    }
    run("empty fixed-level insertion supports the highest permitted level", [] {
        const HnswGraph empty(2, {}, std::nullopt, {2, 2});
        const auto graph = insert_at_level(empty, {10, {0, 0}}, {2, 2, 1}, 32);
        require(graph.entry_node_index() == 0 && graph.node(0).neighbors.size() == 33,
                "first tall point differs");
        for (const auto& list : graph.node(0).neighbors) {
            require(list.empty(), "first node has a self link");
        }
    });
    run("fixed kernel rejects an unsupported new or old level", [] {
        const HnswGraph empty(2, {}, std::nullopt, {2, 2});
        require_throws<std::invalid_argument>([&] {
            (void)insert_at_level(empty, {10, {0, 0}}, {2, 2, 2}, 33);
        });
        const HnswGraph tall(2, {{{10, {0, 0}}, 33,
            std::vector<std::vector<NodeIndex>>(34)}}, 0, {2, 2});
        require_throws<std::invalid_argument>([&] {
            (void)insert_at_level(tall, {20, {1, 0}}, {2, 2, 2}, 0);
        });
    });
    run("a positive controlled level requires M within the upper cap", [] {
        const HnswGraph empty(2, {}, std::nullopt, {2, 1});
        require_throws<std::invalid_argument>([&] {
            (void)insert_at_level(empty, {10, {0, 0}}, {2, 2, 2}, 1);
        });
    });
    run("seeded four-point build has literal links and promotes the entry", [] {
        const auto state = first_three().insert({40, {9, 0}});
        const auto& graph = state.graph();
        require(graph.size() == 4 && graph.entry_node_index() == 3, "seeded entry differs");
        require_links(graph, 0, {{1, 2}});
        require_links(graph, 1, {{0, 2, 3}});
        require_links(graph, 2, {{1, 0, 3}});
        require_links(graph, 3, {{2, 1}, {}, {}});
        const std::array<std::size_t, 4> levels{0, 0, 0, 2};
        for (NodeIndex index = 0; index < graph.size(); ++index) {
            require(graph.node(index).point.id == 10 * (index + 1) && graph.node(index).level == levels[index],
                    "seeded IDs, indices, or levels differ");
        }
        const auto result = hnsw_top_k(graph, std::array<float, 2>{9, 0}, 4, 4);
        const std::array<std::uint64_t, 4> ids{40, 30, 20, 10};
        const std::array<double, 4> distances{0, 16, 25, 81};
        require(result.size() == 4, "full-width query omitted a node");
        for (std::size_t index = 0; index < result.size(); ++index) {
            require(result[index].id == ids[index] && result[index].squared_distance == distances[index],
                    "seeded query differs from hand-calculated exact results");
        }
        std::cout << "Seeded build: levels=[0,0,0,2], entry=40; query=40:0,30:16,20:25,10:81\n";
    });
    run("generator continues across eight accepted insertions", [] {
        HnswBuildState state(2, {8, 8}, {2, 8, 8, 4, 42});
        const std::array<std::size_t, 8> levels{0, 0, 0, 2, 0, 3, 0, 1};
        for (std::size_t index = 0; index < levels.size(); ++index) {
            state = state.insert({index, {static_cast<float>(index), 0}});
            require(state.graph().node(index).level == levels[index], "generator was reset or skipped a draw");
        }
        require(state.graph().entry_node_index() == 5 && state.config().seed == 42,
                "highest entry or initial seed differs");
    });
    run("maximum construction seed retains its literal continuation", [] {
        const auto seed = std::numeric_limits<std::uint64_t>::max();
        HnswBuildState state(2, {8, 8}, {2, 8, 4, 4, seed});
        const std::array<std::size_t, 4> levels{4, 0, 4, 0};
        for (std::size_t index = 0; index < levels.size(); ++index) {
            state = state.insert({index, {static_cast<float>(index), 0}});
            require(state.graph().node(index).level == levels[index],
                    "maximum seed was rejected, truncated, or reset");
        }
        require(state.config().seed == seed && state.graph().entry_node_index() == 0,
                "maximum seed or equal-level entry changed");
    });
    run("same seed and insertion order repeat completed data", [] {
        require(same_graph(first_three().insert({40, {9, 0}}).graph(),
                           first_three().insert({40, {9, 0}}).graph()), "repeat build differs");
    });
    run("late directed-reachability failure preserves graph and next draw", [] {
        require_preserved_after_failure(first_three({2, 2}), {40, {5.5F, 0}});
        std::cout << "Rejected level-2 draft: old triangle and views remain; retry uses level 2\n";
    });
    const std::array<Point, 5> bad_points{{
        {40, {9}}, {10, {9, 0}}, {40, {std::numeric_limits<float>::quiet_NaN(), 0}},
        {40, {std::numeric_limits<float>::infinity(), 0}},
        {40, {0, -std::numeric_limits<float>::infinity()}}
    }};
    for (std::size_t index = 0; index < bad_points.size(); ++index) {
        run("invalid point preserves seed continuation " + std::to_string(index), [&] {
            require_preserved_after_failure(first_three(), bad_points[index]);
        });
    }
    run("empty seeded state still validates point coordinates", [] {
        const HnswBuildState state(2, {2, 2}, {2, 2, 2, 4, 42});
        require_throws<std::invalid_argument>([&] { (void)state.insert({10, {}}); });
        require_throws<std::invalid_argument>([&] {
            (void)state.insert({10, {std::numeric_limits<float>::quiet_NaN(), 0}});
        });
        require(state.graph().size() == 0 && state.insert({10, {0, 0}}).graph().node(0).level == 0,
                "failed first insert changed empty state");
    });
    run("capacity failure preserves the completed state", [] {
        HnswBuildState state(2, {4, 2}, {2, 3, 4, 4, 42});
        for (Point point : std::array<Point, 4>{{{10, {0, 0}}, {20, {4, 0}}, {30, {5, 0}}, {40, {9, 0}}}}) {
            state = state.insert(std::move(point));
        }
        const auto before = state.graph();
        const auto* old_node = &state.graph().node(0);
        require_throws<std::length_error>([&] { (void)state.insert({50, {10, 0}}); });
        require(same_graph(state.graph(), before) && old_node == &state.graph().node(0),
                "capacity failure changed stored state");
    });
    run("preparing a result preserves the original generator and point storage", [] {
        const auto state = first_three();
        const auto before = state.graph();
        Point point{40, {9, 0}};
        const auto prepared = state.insert(point);
        point.values[0] = 999;
        const auto repeat = state.insert({40, {9, 0}});
        require(same_graph(state.graph(), before) && same_graph(prepared.graph(), repeat.graph()) &&
                prepared.graph().node(3).point.values == std::vector<float>{9, 0},
                "preparation changed the original state or aliased point values");
    });
    run("copied and assigned states fork the same generator history", [] {
        const auto state = first_three();
        auto copy = state;
        HnswBuildState assigned(2, {2, 2}, {2, 2, 2, 0, 0});
        assigned = copy;
        copy = copy.insert({40, {9, 0}});
        assigned = assigned.insert({40, {9, 0}});
        require(same_config(assigned.config(), state.config()) && same_graph(copy.graph(), assigned.graph()) &&
                assigned.graph().node(3).level == 2 && state.graph().size() == 3,
                "state copy or assignment lost generator position");
    });
    run("moved state retains continuation and leaves an empty readable graph", [] {
        auto original = first_three();
        auto moved = std::move(original);
        require(original.graph().size() == 0 && !original.graph().entry_node_index(),
                "moved-from graph is not empty");
        require(moved.insert({40, {9, 0}}).graph().node(3).level == 2, "move lost generator history");
    });
    run("completed state owns data after its input state is destroyed", [] {
        const auto result = [] { return first_three().insert({40, {9, 0}}); }();
        require(result.graph().node(0).point.values == std::vector<float>{0, 0}, "old values were lost");
        require_links(result.graph(), 3, {{2, 1}, {}, {}});
    });
    run("cap zero permits seed zero and distinct equal vectors with extreme IDs", [] {
        HnswBuildState state(2, {2, 2}, {2, 2, 2, 0, 0});
        state = state.insert({std::numeric_limits<std::uint64_t>::max(), {0, 0}});
        state = state.insert({0, {0, 0}});
        const auto result = hnsw_top_k(state.graph(), std::array<float, 2>{0, 0}, 2, 2);
        require(state.graph().node(0).level == 0 && state.graph().node(1).level == 0 &&
                result.size() == 2 && result[0].id == 0 &&
                result[1].id == std::numeric_limits<std::uint64_t>::max(),
                "zero seed, zero cap, or source identity failed");
    });
    run("maximum size requests allocate from actual data", [] {
        const auto maximum = std::numeric_limits<std::size_t>::max();
        HnswBuildState state(2, {maximum, maximum}, {maximum, maximum, maximum, 32, 42});
        state = state.insert({10, {0, 0}});
        state = state.insert({20, {1, 0}});
        require(state.graph().size() == 2, "large configuration caused request-sized allocation");
        require_links(state.graph(), 0, {{1}});
        require_links(state.graph(), 1, {{0}});
    });
    const std::array<HnswBuildConfig, 7> bad_settings{{
        {0, 3, 8, 4, 42}, {1, 3, 8, 4, 42}, {2, 1, 8, 4, 42},
        {2, 3, 0, 4, 42}, {2, 3, 8, 33, 42},
        {2, 3, 8, std::numeric_limits<std::size_t>::max(), 42}, {3, 3, 8, 4, 42}
    }};
    for (std::size_t index = 0; index < bad_settings.size(); ++index) {
        run("invalid fixed build settings " + std::to_string(index), [&] {
            require_throws<std::invalid_argument>([&] { HnswBuildState state(2, {2, 2}, bad_settings[index]); });
        });
    }
    run("build rejects M above only the upper cap", [] {
        require_throws<std::invalid_argument>([] { HnswBuildState state(2, {4, 1}, {2, 3, 8, 4, 42}); });
    });
    run("build inherits positive dimension and cap validation", [] {
        require_throws<std::invalid_argument>([] { HnswBuildState state(0, {2, 2}, {2, 2, 2, 4, 42}); });
        require_throws<std::invalid_argument>([] { HnswBuildState state(2, {0, 2}, {2, 2, 2, 4, 42}); });
    });
    std::cout << "HNSW build checks: " << cases << " cases, " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
