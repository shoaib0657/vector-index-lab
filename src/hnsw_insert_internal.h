#pragma once

#include "vector_search/hnsw_insert.h"

namespace vector_search::detail {

// This kernel links a known level. The seeded state supplies that level.
HnswGraph insert_at_level(const HnswGraph& graph, Point point,
                          Layer0InsertConfig config, std::size_t level);

std::size_t level_from_word(std::uint64_t word, std::size_t neighbors_per_node,
                            std::size_t level_cap);

}  // namespace vector_search::detail
