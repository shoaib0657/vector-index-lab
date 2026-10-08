#include "vector_search/exact_search.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vector_search {

std::vector<Neighbor> exact_top_k(
    std::span<const Point> points,
    std::span<const float> query,
    std::size_t k) {
    if (k == 0 || points.empty()) {
        return {};
    }

    std::vector<Neighbor> neighbors;
    neighbors.reserve(points.size());
    for (const Point& point : points) {
        if (point.values.size() != query.size()) {
            throw std::invalid_argument("point and query dimensions differ");
        }

        // Use double precision to limit the error in the distance sum.
        double squared_distance = 0.0;
        for (std::size_t i = 0; i < query.size(); ++i) {
            if (!std::isfinite(query[i]) || !std::isfinite(point.values[i])) {
                throw std::invalid_argument("coordinates must be finite");
            }
            const double difference =
                static_cast<double>(query[i]) - static_cast<double>(point.values[i]);
            squared_distance += difference * difference;
        }
        neighbors.push_back({point.id, squared_distance});
    }

    // Distance calculations take O(Nd) time. The full sort takes O(N log N) time.
    // The neighbor array needs O(N) extra space.
    // Use a full sort to keep this reference implementation easy to check.
    // Use the source ID to give equal distances a fixed order.
    std::sort(neighbors.begin(), neighbors.end(), [](const Neighbor& left, const Neighbor& right) {
        if (left.squared_distance != right.squared_distance) {
            return left.squared_distance < right.squared_distance;
        }
        return left.id < right.id;
    });
    neighbors.resize(std::min(k, neighbors.size()));
    return neighbors;
}

}  // namespace vector_search
