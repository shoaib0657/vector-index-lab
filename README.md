# HNSW Vector Search

A C++20 project for nearest-neighbor search. Its exact squared-L2 search checks every vector, giving us a reliable reference for exploring graph-based search.

## Build and test

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Exact search

Pass `exact_top_k` points with unique IDs, a query, and the number of neighbors you want:

```cpp
#include "vector_search/exact_search.h"

#include <array>
#include <vector>

int main() {
    const std::vector<vector_search::Point> points{
        {8, {1.0F, 0.0F}},
        {3, {0.0F, 0.0F}},
    };
    const std::array<float, 2> query{0.0F, 0.0F};
    const auto neighbors = vector_search::exact_top_k(points, query, 1);
    // ID 3 is the closest point, at squared distance 0.
}
```

Results are ordered by squared distance, then ID when distances tie. Points need the same dimension as the query and finite coordinates; mismatched dimensions or non-finite values throw `std::invalid_argument`. An empty collection or `k = 0` returns no results. If `k` is larger than the collection, every point is returned.

## Vector files and exact runner

Build and test requires a C++20 compiler, CMake 3.20 or later, and Python 3.
The CLI tests use only the Python standard library.

Both base and query files use text format v1. The header gives the row count
and dimension. Each record has one integer source ID and that many finite
float32 coordinates. IDs must be distinct within each file. The loader keeps
the file order and rejects incomplete records, extra fields, and extra records.

For example:

```text
3 2
8 1 0
2 0 1
5 0 0
```

For query `(0, 0)`, exact top-2 returns ID 5 at squared distance 0, then ID 2
at squared distance 1. ID 2 wins the tie with ID 8.

The runner requires nonempty inputs, equal dimensions, and `1 <= k <= base_size`.
This gives exactly `k` neighbors for every query. The library API retains its
more general empty-input and `k > base_size` behavior.

### Prepare the SIFT samples

Download the public [SIFT-128 HDF5 file](https://ann-benchmarks.com/sift-128-euclidean.hdf5)
to `data/sift-128-euclidean.hdf5`. The pinned data packages require Python 3.12
or later. The [ANN Benchmarks dataset table](https://github.com/erikbern/ann-benchmarks#data-sets)
describes the source. Then run:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r scripts/requirements-data.txt
.venv/bin/python -B scripts/test_prepare_sift.py -v
.venv/bin/python -B scripts/prepare_sift.py \
    --source data/sift-128-euclidean.hdf5 \
    --output data/sift-samples-v1 --base-seed 42 --query-seed 43
```

The exporter uses PCG64 seed 42 to select 50,000 `train` rows. The 10,000-row
base is the prefix of that same saved order. An independent PCG64 seed 43
selects 100 `test` queries for both bases. IDs are the original zero-based
source row numbers; base IDs and query IDs have separate namespaces.
Coordinates stay float32 without normalization. The exporter verifies all
written values, saves the full selection and hashes, and preserves matching
existing outputs. It rejects conflicting outputs.

Recompute exact neighbors within each sampled base. The source's published
million-row neighbors do not establish the answer for either sample.

### Run the exact baseline

Use a Release build and new output paths for each run:

```sh
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --parallel 2
ctest --test-dir build/release --output-on-failure
mkdir -p data/exact-baseline/run-1
./build/release/exact_baseline \
    --base data/sift-samples-v1/base_10k.txt \
    --queries data/sift-samples-v1/queries_100.txt \
    --k 10 --warmup 20 --passes 3 \
    --results data/exact-baseline/run-1/top10-10k.csv \
    --timings data/exact-baseline/run-1/timings-10k.csv
```

Use `base_50k.txt` with separate output paths for the larger base. The runner
preserves existing output files and requires their parent directories to exist.
`--help` lists the options. Defaults are `k=10`, 20 warm-up searches,
and three timed passes. Warm-up searches use queries in file order and cycle
from the start if needed.

The results CSV contains
`base_size,dimension,k,query_id,rank,neighbor_id,squared_distance`.
Ranks start at 1. Results use double squared-L2 distances and the distance-then-ID
order. They appear once per query, independently of the number of timed passes.
Later passes must reproduce the first pass's IDs and distances.

The timing CSV contains `pass,query_id,elapsed_us`.
Each entry measures one complete `exact_top_k` call with
`std::chrono::steady_clock` on one search thread. That call includes
validation, distance calculations, full sort, and result allocation.
File loading, warm-up searches, result checks, and CSV output are outside the
measured interval. Answers can match byte for byte across runs; elapsed times
can vary. With 100 queries and three passes, the timing file has 300 records.

The baseline uses the existing full-sort implementation. Save machine/compiler
details, build flags, source revision and dirty state, input hashes, and exact
commands with each measured run before comparing performance.

### Observed exact baseline

The fixed SIFT samples were measured on 2026-10-07. Each row contains 300
search calls: three passes over the same 100 queries after 20 warm-up searches.
Both runs used one search thread, a Release build, and the full-sort exact search.

| Base rows | Run | Median (ms) | p95 (ms) | Derived search rate (queries/s) |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 1 | 6.092418 | 7.339102 | 156.367 |
| 10,000 | 2 | 6.017381 | 7.216956 | 161.310 |
| 50,000 | 1 | 31.653852 | 35.680436 | 31.143 |
| 50,000 | 2 | 31.726000 | 36.259241 | 31.104 |

p95 uses nearest rank: the 285th of 300 sorted times. The derived rate is
`300 / sum_of_measured_search_seconds`; it excludes other process work.
The timer covers coordinate checks, distance calculations, the full sort,
and result allocation. Loading, warm-up, answer comparison, and CSV writing
are outside the interval.

Machine: Intel Core i5-11320H, eight visible logical CPUs, Ubuntu 24.04.5 LTS
under WSL2 2.6.3.0. GCC 13.3.0 and CMake 3.28.3 built the targets with
`-O3 -DNDEBUG -std=c++20`. CPU affinity was not pinned.

All 100 top-10 queries for each base matched an independent NumPy calculation
of double squared distances with distance-then-source-ID ordering.
The answer files also matched byte for byte between runs for each base.

The measured checkout was dirty at `9fa80b0`; that commit alone does not
contain the loader and runner used here. The [measurement metadata](benchmarks/exact-sift-128.json)
records the complete source-file hashes and executable hash.
The [baseline CSV](benchmarks/exact-sift-128.csv) retains all four rows and their
unrounded statistics. See the [benchmark protocol](benchmarks/README.md)
for definitions, input identities, checks, and repeat commands.

## Static HNSW graph

`HnswGraph` owns a completed graph. Each `HnswNode` contains a
`Point`, a highest level, and neighbor lists ordered from layer 0 upward.
Links use `NodeIndex` positions in the node array; `Point::id` retains the
source identity for result output. Keep node order fixed while links use it.

For example, a two-node directed graph can be constructed as follows:

```cpp
#include "vector_search/hnsw_graph.h"

int main() {
    const vector_search::HnswGraph graph(
        2,
        {{{10, {1.0F, 0.0F}}, 0, {{1}}},
         {{40, {0.0F, 0.0F}}, 0, {{}}}},
        0,
        {1, 1});
    // Neighbor index 1 identifies the point with source ID 40.
    const auto neighbor_index = graph.neighbors(0, 0).front();
    const auto source_id = graph.node(neighbor_index).point.id;
    return source_id == 40 ? 0 : 1;
}
```

Construction rejects invalid snapshots with `std::invalid_argument`: zero
dimension or degree limits, duplicate source IDs, unequal dimensions,
non-finite coordinates, inconsistent level/list counts, invalid entry,
invalid or repeated links, self-links, targets absent from a layer, excess
degree, and nodes unreachable from the entry through layer-0 directed links.
Base and upper layers have separate degree limits. Reciprocal links are
allowed, and directed graphs are also valid when every node is reachable.

An empty graph requires a positive dimension, positive limits, and no entry.
Equal coordinates with different source IDs remain distinct points.
The stored entry must be at the highest populated level.

`node(node_index)` and `neighbors(node_index, layer)` return const references
or spans. Invalid indices and absent layers throw `std::out_of_range`.
Keep the graph alive and its storage unchanged while using a view.
Acquire new views after moving or assigning a graph. A moved-from graph
is empty, has no entry, and retains its positive dimension and degree limits.

This module represents and validates a completed static graph. Search,
insertion, and pruning are separate stages. Candidates and visited state
belong to a query call.

## HNSW query search

`hnsw_top_k` reads a validated static graph. It uses a nearest-first candidate
heap, a farthest-first retained best heap, and fresh visited flags for each
layer call. Upper layers use `ef = 1`; layer zero uses `ef_search`.

```cpp
#include "vector_search/hnsw_search.h"

#include <array>
#include <iostream>

int main() {
    const vector_search::HnswGraph graph(
        2,
        {{{10, {1.F, 0.F}}, 0, {{1}}}, {{40, {0.F, 0.F}}, 0, {{}}}},
        0,
        {1, 1});
    const std::array<float, 2> query{0.F, 0.F};
    vector_search::HnswSearchTrace trace;
    const auto result = vector_search::hnsw_top_k(graph, query, 1, 2, &trace);
    if (result.size() != 1) {
        return 1;
    }
    std::cout << result[0].id << ':' << result[0].squared_distance << '\n';
    return result[0].id == 40 && result[0].squared_distance == 0 ? 0 : 1;
}
```

This program prints `40:0`. Returned neighbors carry source IDs and double
squared-L2 distances, sorted by distance then source ID. For a nonempty graph
and positive `k`, the query must match the graph dimension, contain finite
coordinates, and use `ef_search >= k`. Bad inputs throw `std::invalid_argument`.
Zero `k` or an empty graph returns no neighbors before these checks.
Requests above the graph size return all nodes when `ef_search >= k`.
When `ef_search >= graph.size()`, layer zero starts at the stored entry.
This preserves the graph's validated reachability even if upper descent
reaches a sink in a directed graph. Narrow searches use the descended entry.
Scratch allocation follows graph size and admitted nodes, not the requested
`k` or `ef_search`.

Source IDs order equal-distance heap items and retained results. Admission
uses a strictly smaller distance when the best heap is full. Stopping uses
`candidate_distance > farthest_retained_distance`, so an equal-distance
candidate is still expanded. An equal-distance point that was not admitted
cannot appear in the returned results.

Pass no trace pointer for ordinary queries. A requested trace owns layer
entries, visited IDs in discovery order, expanded IDs, sorted retained
neighbors, and any candidate that caused stopping. It is replaced after a
successful call, cleared by an early empty result, and unchanged on an
exception. The graph must stay alive and unchanged throughout the call.
The graph stores no mutable query scratch state; callers use separate trace
objects for separate queries.

This search reads completed snapshots. The insertion APIs below build
level-zero or seeded multi-layer snapshots. Construction uses the same
internal search with all retained entries from the previous layer.
The tiny trace tests establish search behavior, not recall or speed on SIFT.

## Layer-zero insertion

`insert_layer0` returns a new validated, owned `HnswGraph`. It reads the input
through a const reference and prepares changes in a local draft. Replace the
caller graph only after a successful return:

```cpp
#include "vector_search/hnsw_insert.h"

int main() {
    vector_search::HnswGraph graph(2, {}, std::nullopt, {1, 1});
    const vector_search::Layer0InsertConfig config{1, 3, 3};
    graph = vector_search::insert_layer0(graph, {10, {0, 0}}, config);
    graph = vector_search::insert_layer0(graph, {20, {4, 0}}, config);
    graph = vector_search::insert_layer0(graph, {30, {5, 0}}, config);
    // Entry 10 reaches every node through 10 -> 20 -> 30.
    return graph.size() == 3 && graph.neighbors(1, 0).front() == 2 ? 0 : 1;
}
```

The configuration is `{neighbors_per_node, ef_construction, max_nodes}`.
Require a positive selection count, `ef_construction >= neighbors_per_node`,
a positive node limit, and a selection count no larger than the graph's base
cap. `degree_limits()` returns the original base and upper caps. Insertion
preserves both caps, dimension, existing node indices, and the stored entry.
The first point in an empty graph becomes entry index 0.

Points must have the graph dimension, finite coordinates, and a source ID
not already stored. Identical vectors with distinct IDs are allowed. Point
is passed by value: an lvalue is copied, while an explicitly moved argument
can be consumed even if insertion fails.

For nonempty input, search the old graph with width `ef_construction` and
select up to `neighbors_per_node` candidates in distance-then-source-ID order.
The shared search retains stored indices as well as source IDs. Append the
new node and reverse links using those indices. An old list that exceeds the base cap keeps its nearest entries,
measured from that list's owner using double squared L2. Source IDs break
ties. Pruning changes only that owner's list, so final links can be directed.
Unpruned old lists retain their order.

Invalid settings, point data, or multi-layer input throw `std::invalid_argument`.
So does a draft that fails final graph validation. Simple nearest-neighbor
pruning can disconnect nodes even for a valid point: adding `(0.5, 0)` to
`10:(0,0) <-> 20:(4,0)` with base cap 1 retains `10 <-> 30` and disconnects
20 from entry 10. This draft is rejected. The original graph, its queries,
and its borrowed views stay valid until the caller moves or assigns it.
Connectivity repair is outside this API.

A graph at or above `max_nodes` throws `std::length_error`; this logical limit
is separate from container capacity. Container and allocation failures
propagate. Storage follows actual nodes, edges, and admitted candidates;
large node limits or search widths do not trigger request-sized reservations.
Snapshot copying costs `O(Nd + E)` for N points of dimension d and E links,
in addition to search, pruning, and validation.

This API accepts only level-zero graphs and uses no random generator. It is
a small construction baseline, not full multi-layer HNSW. Graph assignment
is ordinary single-threaded replacement, not concurrent publication. Tiny
insertion checks establish correctness for the stated contract; they do not
measure SIFT recall or construction speed.

## Seeded multi-layer construction

`HnswBuildState` owns a completed graph, fixed configuration, and a private
`std::mt19937_64` generator. Its const `insert(Point)` prepares a new state.
Assign that result after success to advance both the graph and generator:

```cpp
#include "vector_search/hnsw_insert.h"
#include "vector_search/hnsw_search.h"

#include <array>
#include <iostream>

int main() {
    vector_search::HnswBuildState state(2, {4, 2}, {2, 3, 4, 4, 42});
    for (auto point : std::array<vector_search::Point, 4>{{
             {10, {0, 0}}, {20, {4, 0}}, {30, {5, 0}}, {40, {9, 0}}}}) {
        state = state.insert(point);
    }
    const auto& graph = state.graph();
    std::cout << "entry=" << graph.node(graph.entry_node_index().value()).point.id << '\n';
    for (const auto& neighbor : vector_search::hnsw_top_k(
             graph, std::array<float, 2>{9, 0}, 4, 4)) {
        std::cout << neighbor.id << ':' << neighbor.squared_distance << '\n';
    }
    return graph.node(3).level == 2 ? 0 : 1;
}
```

This example prints entry 40 followed by `40:0`, `30:16`, `20:25`, and
`10:81`. Its levels are [0,0,0,2]. Configuration fields are
`{neighbors_per_node, ef_construction, max_nodes, level_cap, seed}`.
Require M>=2, M within both degree caps, efConstruction>=M, a positive node
limit, and level_cap in [0,32]. Seed zero is valid. The constructor creates
an empty graph with the supplied positive dimension and degree limits.

Initialize the engine once from the seed. For each successful insertion,
use one word and map `U = ((word >> 11) + 1) * 2^-53`, then take
`floor(-log(U)/log(M))`, capped before conversion to an integer. This requires
binary64 double. U is in (0,1], so the logarithm never receives zero.
Cap zero forces level zero and still consumes one word on success. Copying
a state forks its construction history; the configuration's seed remains
the initial seed. Identical settings, seed, and successful insertion order
repeat on the same build/runtime. Logarithm thresholds and floating
calculations limit claims of identical graphs across different runtimes.

Above the new level, descend with one entry and width 1. On shared layers,
search with efConstruction, select up to M nearest candidates, add reverse
links, and prune overfull lists from each owner's vector. Carry the entire
retained W into the next lower layer, including candidates not selected as
neighbors. Search reads the old graph; partial links stay in a local draft.
New layers above the old top start empty. Promote the stored entry only
when the new level exceeds the old top level.

Final graph validation can reject a draft whose pruning or entry promotion
breaks directed base reachability. This failure preserves the original
graph, configuration, generator position, and borrowed views. A retry uses
the same next draw. Invalid point/configuration data throws
`std::invalid_argument`; logical capacity throws `std::length_error`.
Container and allocation failures propagate. A moved point argument can
be consumed on failure. Reacquire graph views after assigning or moving the
state. A moved-from state has an empty readable graph; construct a new state
to restart from its seed.

This small snapshot builder uses simple nearest-neighbor selection from
Algorithm 3. It copies and validates the graph per insertion, which limits
its suitability for larger builds. Diverse selection, connectivity repair,
deletion, and concurrent construction/publication are outside this API.
The multi-layer examples establish the stated behavior; SIFT construction,
recall, and construction performance have not been measured.
