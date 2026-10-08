#include "vector_search/vector_text.h"

#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expect_invalid(std::string_view contents) {
    std::istringstream input{std::string(contents)};
    try {
        (void)vector_search::load_vector_text(input);
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("malformed input was accepted");
}

struct FixtureFile {
    std::filesystem::path path = std::filesystem::current_path() /
        ("vector-text-fixture-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".txt");
    FixtureFile() {
        std::ofstream output(path);
        output << "1 2\n9 3 4\n";
        require(static_cast<bool>(output), "fixture file could not be written");
    }
    ~FixtureFile() {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }
};

}  // namespace

int main() {
    int cases = 0;
    int failures = 0;
    const auto run = [&](std::string_view name, auto check) {
        ++cases;
        try {
            check();
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        }
    };

    run("source IDs, row order, coordinates, and exact tie cutoff", [] {
        std::istringstream input("3 2\n8 1 0\n2 0 1\n5 0 0\n");
        const auto data = vector_search::load_vector_text(input);
        require(data.dimension == 2 && data.points.size() == 3, "header or row count changed");
        require(data.points[0].id == 8 && data.points[1].id == 2 && data.points[2].id == 5, "source IDs or row order changed");
        require(data.points[0].values == std::vector<float>{1, 0} &&
                data.points[1].values == std::vector<float>{0, 1} &&
                data.points[2].values == std::vector<float>{0, 0}, "coordinates became attached to the wrong IDs");
        const std::vector<float> query{0, 0};
        // IDs 2 and 8 have equal distances. ID 2 must win at the cutoff.
        const auto answer = vector_search::exact_top_k(data.points, query, 2);
        require(answer.size() == 2 && answer[0].id == 5 && answer[0].squared_distance == 0 &&
                answer[1].id == 2 && answer[1].squared_distance == 1, "loaded fixture produced the wrong exact top-2");
    });
    run("full uint64 ID and float32 precision", [] {
        std::istringstream input("1 2\n18446744073709551615 0.123456791 1.00000012\n");
        const auto data = vector_search::load_vector_text(input);
        require(data.points.size() == 1, "precision fixture row missing");
        require(data.points[0].id == std::numeric_limits<std::uint64_t>::max(), "ID precision was lost");
        require(data.points[0].values == std::vector<float>{0.12345679F, std::nextafter(1.0F, 2.0F)}, "float32 precision was lost");
    });
    run("zero-row dataset retains its dimension", [] {
        std::istringstream input("0 2\n");
        const auto data = vector_search::load_vector_text(input);
        require(data.dimension == 2 && data.points.empty(), "zero-row header was not preserved");
    });
    run("duplicate coordinates with distinct IDs remain distinct records", [] {
        std::istringstream input("2 2\n8 1 0\n2 1 0\n");
        const auto data = vector_search::load_vector_text(input);
        require(data.points.size() == 2 && data.points[0].id == 8 && data.points[1].id == 2, "valid duplicate coordinates were merged");
    });
    run("file wrapper reads the same record", [] {
        const FixtureFile fixture;
        const auto data = vector_search::load_vector_text_file(fixture.path);
        require(data.dimension == 2 && data.points.size() == 1, "file wrapper lost a record");
        require(data.points[0].id == 9 && data.points[0].values == std::vector<float>{3, 4}, "file wrapper changed the record");
    });

    const std::array<std::pair<std::string_view, std::string_view>, 15> malformed{{
        {"missing header", ""},
        {"fractional header", "1.5 2\n"},
        {"overflowing header", "18446744073709551616 2\n"},
        {"zero dimension", "1 0\n8\n"},
        {"extra header field", "1 2 extra\n8 1 0\n"},
        {"missing record", "1 2\n"},
        {"missing coordinate", "1 2\n8 1\n"},
        {"extra coordinate", "1 2\n8 1 0 7\n"},
        {"extra record", "1 2\n8 1 0\n2 0 1\n"},
        {"duplicate ID", "2 2\n8 1 0\n8 0 1\n"},
        {"negative ID", "1 2\n-1 1 0\n"},
        {"fractional ID", "1 2\n1.5 1 0\n"},
        {"nonfinite coordinate", "1 2\n8 nan 0\n"},
        {"overflowing coordinate", "1 2\n8 3.5e99 0\n"},
        {"partial coordinate token", "1 2\n8 1x 0\n"},
    }};
    for (const auto& [name, contents] : malformed) {
        run(name, [&] { expect_invalid(contents); });
    }
    std::cout << "Vector text checks: " << cases << " cases, " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
