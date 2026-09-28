#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace vector_search {

struct Point {
    std::uint64_t id;
    std::vector<float> values;
};

struct Neighbor {
    std::uint64_t id;
    double squared_distance;
};

// Brute-force search. We score every point, sort by squared L2 (ID breaks a
// tie), and keep the first k. IDs should be unique. With k == 0 or no points,
// we return before checking the vectors; otherwise bad dimensions or NaN/Inf
// values throw. Asking for more than we have returns everything.
std::vector<Neighbor> exact_top_k(
    std::span<const Point> points,
    std::span<const float> query,
    std::size_t k);

}  // namespace vector_search
