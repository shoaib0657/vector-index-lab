#include "vector_search/exact_search.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

int main() {
    const std::vector<vector_search::Point> points{
        {10, {0.0F, 0.0F}},
        {4, {2.0F, 0.0F}},
        {7, {0.0F, 0.0F}},
        {2, {1.0F, 2.0F}},
        {9, {3.0F, 1.0F}},
    };
    const std::array<float, 2> query{1.0F, 1.0F};

    const auto tied = vector_search::exact_top_k(points, query, 4);
    const std::array<std::uint64_t, 4> expected_ids{2, 4, 7, 10};
    const std::array<double, 4> expected_distances{1.0, 2.0, 2.0, 2.0};
    if (tied.size() != expected_ids.size()) {
        std::cerr << "Duplicate-coordinate tie: wrong result count\n";
        return 1;
    }
    for (std::size_t i = 0; i < tied.size(); ++i) {
        if (tied[i].id != expected_ids[i] ||
            tied[i].squared_distance != expected_distances[i]) {
            std::cerr << "Duplicate-coordinate tie: wrong result at rank " << i << '\n';
            return 1;
        }
    }

    if (!vector_search::exact_top_k(points, query, 0).empty()) {
        std::cerr << "k=0 should return no results\n";
        return 1;
    }
    if (vector_search::exact_top_k(points, query, 99).size() != points.size()) {
        std::cerr << "k>N should return all points\n";
        return 1;
    }
    const std::vector<vector_search::Point> empty;
    if (!vector_search::exact_top_k(empty, query, 3).empty()) {
        std::cerr << "Empty input should return no results\n";
        return 1;
    }

    bool wrong_dimension_rejected = false;
    try {
        const std::vector<vector_search::Point> wrong_dimension{{1, {1.0F}}};
        (void)vector_search::exact_top_k(wrong_dimension, query, 1);
    } catch (const std::invalid_argument&) {
        wrong_dimension_rejected = true;
    }
    if (!wrong_dimension_rejected) {
        std::cerr << "Wrong dimension should be rejected\n";
        return 1;
    }

    bool non_finite_query_rejected = false;
    try {
        const std::array<float, 2> nan_query{
            std::numeric_limits<float>::quiet_NaN(), 0.0F};
        (void)vector_search::exact_top_k(points, nan_query, 1);
    } catch (const std::invalid_argument&) {
        non_finite_query_rejected = true;
    }
    if (!non_finite_query_rejected) {
        std::cerr << "Non-finite query coordinate should be rejected\n";
        return 1;
    }

    bool non_finite_point_rejected = false;
    try {
        const std::vector<vector_search::Point> infinite_point{
            {1, {std::numeric_limits<float>::infinity(), 0.0F}}};
        (void)vector_search::exact_top_k(infinite_point, query, 1);
    } catch (const std::invalid_argument&) {
        non_finite_point_rejected = true;
    }
    if (!non_finite_point_rejected) {
        std::cerr << "Non-finite stored coordinate should be rejected\n";
        return 1;
    }

    return 0;
}
