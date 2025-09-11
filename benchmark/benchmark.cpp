#include <iostream>
#include <string>
#include <vector>
#include <cstdio>    // for popen, pclose
#include <regex>     // for parsing

int main() {
    const int runs = 15;
    std::vector<long long> cpu_times;
    std::vector<long long> gpu_times;

    for (int i = 0; i < runs; i++) {
        std::cout << "Run " << (i + 1) << "..." << std::endl;

        // Run main.exe and capture its stdout
        FILE* pipe = _popen("main.exe", "r");
        if (!pipe) {
            std::cerr << "Failed to run main.exe" << std::endl;
            return 1;
        }

        std::string output;
        char buffer[256];
        while (fgets(buffer, sizeof(buffer), pipe)) {
            output += buffer;
        }
        _pclose(pipe);

        // Parse using regex
        std::regex cpu_regex(R"(cpu execution: (\d+))");
        std::regex gpu_regex(R"(gpu execution: (\d+))");

        std::smatch match;
        if (std::regex_search(output, match, cpu_regex)) {
            cpu_times.push_back(std::stoll(match[1]));
        }
        if (std::regex_search(output, match, gpu_regex)) {
            gpu_times.push_back(std::stoll(match[1]));
        }

        std::cout << output << std::endl;
    }

    // Compute averages
    auto avg = [](const std::vector<long long>& v) {
        long long sum = 0;
        for (auto t : v) sum += t;
        return (double)sum / v.size();
    };

    std::cout << "\n==== Benchmark Results (average over " << runs << " runs) ====\n";
    std::cout << "Average CPU time: " << avg(cpu_times) << " ms\n";
    std::cout << "Average GPU time: " << avg(gpu_times) << " ms\n";
    std::cout << "Average speedup: " << avg(cpu_times) / avg(gpu_times) << "x\n";

    return 0;
}
