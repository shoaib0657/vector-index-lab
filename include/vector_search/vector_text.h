#pragma once

#include "vector_search/exact_search.h"

#include <filesystem>
#include <istream>

namespace vector_search {

// This data set owns its points and coordinates.
// Keep it alive while a span refers to this data.
struct VectorDataset {
    std::size_t dimension;
    std::vector<Point> points;
};

// Text v1 starts with "row_count dimension".
// Each record has one source ID and exactly dimension finite float coordinates.
// Use distinct IDs. Keep the records in file order.
// A zero row count is valid. The dimension must be positive.
VectorDataset load_vector_text(std::istream& input);
VectorDataset load_vector_text_file(const std::filesystem::path& path);

}  // namespace vector_search
