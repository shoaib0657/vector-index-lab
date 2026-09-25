#include <array>
#include <iostream>
#include <numeric>
#include <span>

int main() {
    constexpr std::array numbers{1, 2, 3};
    const std::span<const int> values{numbers};
    const int sum = std::accumulate(values.begin(), values.end(), 0);

    if (sum != 6) {
        std::cerr << "Unexpected sum: " << sum << '\n';
        return 1;
    }

    std::cout << "C++20 span sum: " << sum << '\n';
    return 0;
}
