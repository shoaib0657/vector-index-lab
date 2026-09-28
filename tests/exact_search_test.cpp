#include "vector_search/exact_search.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    const std::vector<vector_search::Point> points{
        {8, {0.0F, 1.0F}},
        {3, {2.0F, 2.0F}},
        {6, {3.0F, 1.0F}},
        {1, {1.0F, 1.0F}},
        {9, {2.0F, 1.0F}},
        {5, {4.0F, 2.0F}},
    };
    const std::array<float, 2> query{2.0F, 1.0F};
    const auto results = vector_search::exact_top_k(points, query, 3);

    const std::array<std::uint64_t, 3> expected_ids{9, 1, 3};
    const std::array<double, 3> expected_distances{0.0, 1.0, 1.0};
    if (results.size() != expected_ids.size()) {
        std::cerr << "Expected three neighbors, got " << results.size() << '\n';
        return 1;
    }
    for (std::size_t i = 0; i < results.size(); ++i) {
        if (results[i].id != expected_ids[i] ||
            results[i].squared_distance != expected_distances[i]) {
            std::cerr << "Wrong neighbor at rank " << i << '\n';
            return 1;
        }
    }
    return 0;
}
