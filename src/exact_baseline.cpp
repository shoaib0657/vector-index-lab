#include "vector_search/vector_text.h"

#include <charconv>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

struct Options {
    std::filesystem::path base;
    std::filesystem::path queries;
    std::filesystem::path results;
    std::filesystem::path timings;
    std::size_t k;
    std::size_t warmup;
    std::size_t passes;
};

void print_help() {
    std::cout
        << "Usage: exact_baseline --base FILE --queries FILE --results CSV --timings CSV\n"
        << "       [--k 10] [--warmup 20] [--passes 3]\n"
        << "Inputs use source-id-float32-text-v1. Outputs must be distinct new files\n"
        << "in existing directories. k and passes must be positive; k cannot exceed\n"
        << "the base size. Both inputs must be nonempty and have equal dimensions.\n"
        << "Warmup cycles through queries in file order, then each timed pass visits\n"
        << "every query in that order. One search thread; load/write time excluded.\n";
}

std::size_t parse_count(const std::string& text, const std::string& flag) {
    std::size_t value{};
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
        throw std::invalid_argument("Invalid unsigned integer for " + flag + ": " + text);
    }
    return value;
}

Options parse_options(int argc, char** argv) {
    std::map<std::string, std::string> values{
        {"--base", ""}, {"--queries", ""}, {"--results", ""}, {"--timings", ""},
        {"--k", "10"}, {"--warmup", "20"}, {"--passes", "3"},
    };
    std::set<std::string> seen;
    for (int index = 1; index < argc; ++index) {
        const std::string flag = argv[index];
        const auto field = values.find(flag);
        if (field == values.end()) {
            throw std::invalid_argument("Unknown option: " + flag);
        }
        if (!seen.insert(flag).second) {
            throw std::invalid_argument("Duplicate option: " + flag);
        }
        if (++index == argc) {
            throw std::invalid_argument("Missing value for " + flag);
        }
        field->second = argv[index];
    }
    for (const auto* flag : {"--base", "--queries", "--results", "--timings"}) {
        if (values.at(flag).empty()) {
            throw std::invalid_argument(std::string("Required option: ") + flag);
        }
    }
    Options options{
        values.at("--base"), values.at("--queries"),
        values.at("--results"), values.at("--timings"),
        parse_count(values.at("--k"), "--k"),
        parse_count(values.at("--warmup"), "--warmup"),
        parse_count(values.at("--passes"), "--passes"),
    };
    if (options.k == 0 || options.passes == 0) {
        throw std::invalid_argument("k and passes must be positive");
    }
    return options;
}

void check_destination(const std::filesystem::path& path) {
    if (std::filesystem::exists(path) || std::filesystem::is_symlink(path)) {
        throw std::invalid_argument("Preserving existing output: " + path.string());
    }
    const auto parent = path.has_parent_path() ? path.parent_path() : std::filesystem::path(".");
    if (!std::filesystem::is_directory(parent)) {
        throw std::invalid_argument("Output directory does not exist: " + parent.string());
    }
}

bool same_neighbors(
    const std::vector<vector_search::Neighbor>& left,
    const std::vector<vector_search::Neighbor>& right) {
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (left[index].id != right[index].id ||
            left[index].squared_distance != right[index].squared_distance) {
            return false;
        }
    }
    return true;
}

void run(const Options& options) {
    // The data sets own the memory used by the spans in exact_top_k.
    const auto base = vector_search::load_vector_text_file(options.base);
    const auto queries = vector_search::load_vector_text_file(options.queries);
    if (base.dimension != queries.dimension) {
        throw std::invalid_argument("Base and query dimensions differ");
    }
    if (base.points.empty() || queries.points.empty()) {
        throw std::invalid_argument("Base and queries must be nonempty");
    }
    if (options.k > base.points.size()) {
        throw std::invalid_argument("k cannot exceed the base size");
    }
    // Preserve earlier evidence. Check both output paths before the first search.
    check_destination(options.results);
    check_destination(options.timings);
    if (std::filesystem::weakly_canonical(options.results) ==
        std::filesystem::weakly_canonical(options.timings)) {
        throw std::invalid_argument("Results and timings need distinct output files");
    }

    // Use the first queries to warm the search path.
    // Repeat from the start if the query set is shorter.
    for (std::size_t index = 0; index < options.warmup; ++index) {
        (void)vector_search::exact_top_k(
            base.points, queries.points[index % queries.points.size()].values, options.k);
    }

    std::vector<std::vector<vector_search::Neighbor>> answers(queries.points.size());
    std::vector<std::vector<double>> elapsed(
        options.passes, std::vector<double>(queries.points.size()));
    for (std::size_t pass = 0; pass < options.passes; ++pass) {
        for (std::size_t query = 0; query < queries.points.size(); ++query) {
            // Measure the full search call. Keep file I/O outside this interval.
            const auto start = std::chrono::steady_clock::now();
            auto neighbors = vector_search::exact_top_k(
                base.points, queries.points[query].values, options.k);
            const auto finish = std::chrono::steady_clock::now();
            elapsed[pass][query] =
                std::chrono::duration<double, std::micro>(finish - start).count();
            // Keep the result check outside the measured interval.
            if (pass == 0) {
                answers[query] = std::move(neighbors);
            } else if (!same_neighbors(answers[query], neighbors)) {
                throw std::runtime_error("Exact results changed between timed passes");
            }
        }
    }

    // Write the CSV files after all searches.
    // Separate answers from elapsed times so we can compare answers across runs.
    std::ofstream results(options.results);
    std::ofstream timings(options.timings);
    try {
        results.exceptions(std::ios::badbit | std::ios::failbit);
        timings.exceptions(std::ios::badbit | std::ios::failbit);
        results.imbue(std::locale::classic());
        timings.imbue(std::locale::classic());
        // Use enough decimal digits to recover each double value from the CSV file.
        results << std::setprecision(std::numeric_limits<double>::max_digits10);
        timings << std::setprecision(std::numeric_limits<double>::max_digits10);
        results << "base_size,dimension,k,query_id,rank,neighbor_id,squared_distance\n";
        for (std::size_t query = 0; query < queries.points.size(); ++query) {
            for (std::size_t rank = 0; rank < answers[query].size(); ++rank) {
                const auto& neighbor = answers[query][rank];
                results << base.points.size() << ',' << base.dimension << ',' << options.k << ','
                        << queries.points[query].id << ',' << rank + 1 << ','
                        << neighbor.id << ',' << neighbor.squared_distance << '\n';
            }
        }
        timings << "pass,query_id,elapsed_us\n";
        for (std::size_t pass = 0; pass < options.passes; ++pass) {
            for (std::size_t query = 0; query < queries.points.size(); ++query) {
                timings << pass + 1 << ',' << queries.points[query].id << ','
                        << elapsed[pass][query] << '\n';
            }
        }
        results.close();
        timings.close();
    } catch (...) {
        // Remove incomplete output files. Keep failed runs out of the saved evidence.
        results.exceptions(std::ios::goodbit);
        timings.exceptions(std::ios::goodbit);
        results.close();
        timings.close();
        std::error_code ignored;
        std::filesystem::remove(options.results, ignored);
        std::filesystem::remove(options.timings, ignored);
        throw;
    }
    std::cout << "base=" << base.points.size() << " queries=" << queries.points.size()
              << " dimension=" << base.dimension << " k=" << options.k
              << " warmup=" << options.warmup << " passes=" << options.passes
              << " search_threads=1\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        print_help();
        return 0;
    }
    try {
        run(parse_options(argc, argv));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "exact_baseline: " << error.what() << '\n';
        return 1;
    }
}
