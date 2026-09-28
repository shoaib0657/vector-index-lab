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
