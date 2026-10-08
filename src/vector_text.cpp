#include "vector_search/vector_text.h"

#include <charconv>
#include <cmath>
#include <fstream>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_set>
#include <utility>

namespace vector_search {
namespace {

template <typename Number>
Number parse_number(const std::string& token, const char* description) {
    Number value{};
    const auto result = std::from_chars(token.data(), token.data() + token.size(), value);
    // Reject partial tokens. For example, '1x' must not become coordinate 1.
    if (result.ec != std::errc{} || result.ptr != token.data() + token.size()) {
        throw std::invalid_argument(std::string("Invalid ") + description + ": " + token);
    }
    return value;
}

std::string next_token(std::istringstream& line, const char* description) {
    std::string token;
    if (!(line >> token)) {
        throw std::invalid_argument(std::string("Missing ") + description);
    }
    return token;
}

void require_line_end(std::istringstream& line) {
    std::string extra;
    if (line >> extra) {
        throw std::invalid_argument("Unexpected field at the end of a record: " + extra);
    }
}

}  // namespace

VectorDataset load_vector_text(std::istream& input) {
    std::string text;
    if (!std::getline(input, text)) {
        throw std::invalid_argument("Missing vector-file header");
    }
    std::istringstream header(text);
    header.imbue(std::locale::classic());
    const auto count = parse_number<std::size_t>(next_token(header, "row count"), "row count");
    const auto dimension = parse_number<std::size_t>(next_token(header, "dimension"), "dimension");
    require_line_end(header);
    if (dimension == 0) {
        throw std::invalid_argument("Vector dimension must be positive");
    }

    VectorDataset data{dimension, {}};
    data.points.reserve(count);
    std::unordered_set<std::uint64_t> seen_ids;
    for (std::size_t row = 0; row < count; ++row) {
        if (!std::getline(input, text)) {
            throw std::invalid_argument("Missing vector record " + std::to_string(row + 1));
        }
        std::istringstream record(text);
        record.imbue(std::locale::classic());
        Point point{parse_number<std::uint64_t>(next_token(record, "source ID"), "source ID"), {}};
        if (!seen_ids.insert(point.id).second) {
            throw std::invalid_argument("Duplicate source ID: " + std::to_string(point.id));
        }
        point.values.reserve(dimension);
        for (std::size_t coordinate = 0; coordinate < dimension; ++coordinate) {
            const auto value = parse_number<float>(next_token(record, "coordinate"), "coordinate");
            if (!std::isfinite(value)) {
                throw std::invalid_argument("Coordinates must be finite");
            }
            point.values.push_back(value);
        }
        require_line_end(record);
        data.points.push_back(std::move(point));
    }
    // Permit blank lines after the final record. Reject any extra data.
    while (std::getline(input, text)) {
        std::istringstream trailing(text);
        trailing.imbue(std::locale::classic());
        require_line_end(trailing);
    }
    if (input.bad()) {
        throw std::runtime_error("I/O failure while reading vector file");
    }
    return data;
}

VectorDataset load_vector_text_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open vector file: " + path.string());
    }
    return load_vector_text(input);
}

}  // namespace vector_search
