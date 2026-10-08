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

// Score all points with squared L2. Sort by distance, then ID.
// Use unique IDs. If k is zero or the base is empty, return no neighbors.
// Otherwise, unequal dimensions or non-finite coordinates cause an exception.
// If k exceeds the base size, return all points.
std::vector<Neighbor> exact_top_k(
    std::span<const Point> points,
    std::span<const float> query,
    std::size_t k);

}  // namespace vector_search
