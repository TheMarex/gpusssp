#ifndef GPUSSSP_EXPERIMENTS_EXPERIMENT_UTIL_HPP
#define GPUSSSP_EXPERIMENTS_EXPERIMENT_UTIL_HPP

#include "common/string_util.hpp"
#include "queries.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpusssp::experiments
{

inline std::vector<std::string> parse_metrics(const std::string &metrics_str)
{
    std::vector<std::string> metrics;
    common::detail::split(metrics, metrics_str, ",");

    for (auto &metric : metrics)
    {
        if (metric != "time" && metric != "edges_relaxed")
        {
            throw std::runtime_error("unknown metric '" + metric +
                                     "', available: time, edges_relaxed");
        }
    }

    return metrics;
}

inline void validate_metrics(const std::vector<std::string> &metrics)
{
#ifdef ENABLE_STATISTICS
    for (const auto &metric : metrics)
    {
        if (metric == "time")
        {
            throw std::runtime_error("'time' metric not available when ENABLE_STATISTICS is on "
                                     "(overhead corrupts timing)");
        }
    }
#else
    for (const auto &metric : metrics)
    {
        if (metric == "edges_relaxed")
        {
            throw std::runtime_error("'edges_relaxed' metric requires ENABLE_STATISTICS");
        }
    }
#endif
}

inline uint64_t get_unix_timestamp()
{
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::seconds>(duration).count();
}

inline std::string get_git_sha()
{
    std::array<char, 128> buffer;
    std::string result;
    auto pipe_deleter = [](FILE *f)
    {
        if (f)
            pclose(f);
    };
    std::unique_ptr<FILE, decltype(pipe_deleter)> pipe(
        popen("git rev-parse --short HEAD 2>/dev/null", "r"), pipe_deleter);
    if (!pipe)
    {
        return "unknown";
    }
    while (std::fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
    {
        result += buffer.data();
    }
    if (!result.empty() && result.back() == '\n')
    {
        result.pop_back();
    }
    return result.empty() ? "unknown" : result;
}

inline std::string hash_queries_content(const std::vector<Query> &queries)
{
    // Build a string representation of all queries
    std::ostringstream oss;
    for (const auto &query : queries)
    {
        oss << query.from << "," << query.to << ";";
    }

    std::string content = oss.str();

    // Hash the string
    std::hash<std::string> hasher;
    size_t hash_value = hasher(content);

    // Format as 8-character hex string
    std::ostringstream hex_stream;
    hex_stream << std::hex << std::setw(8) << std::setfill('0')
               << (hash_value & 0xFFFFFFFF); // Use lower 32 bits

    return hex_stream.str();
}

inline std::string hash_device_name(const std::string &device_name)
{
    std::hash<std::string> hasher;
    size_t hash_value = hasher(device_name);

    // Format as 8-character hex string
    std::ostringstream hex_stream;
    hex_stream << std::hex << std::setw(8) << std::setfill('0')
               << (hash_value & 0xFFFFFFFF); // Use lower 32 bits

    return hex_stream.str();
}

inline std::string extract_graph_name(const std::string &graph_base_path)
{
    std::filesystem::path path(graph_base_path);
    return path.filename().string();
}

inline void create_experiment_directories(const std::string &xp_base_path,
                                          const std::string &xp_name,
                                          const std::string &graph_name,
                                          const std::string &query_hash,
                                          const std::string &device_id)
{
    std::filesystem::path dir_path =
        std::filesystem::path(xp_base_path) / xp_name / graph_name / query_hash / device_id;
    std::filesystem::create_directories(dir_path);
}

inline std::string generate_experiment_filename(const std::string &xp_base_path,
                                                const std::string &xp_name,
                                                const std::string &graph_name,
                                                const std::string &query_hash,
                                                const std::string &device_id,
                                                uint64_t timestamp,
                                                const std::string &variant)
{
    std::filesystem::path dir_path =
        std::filesystem::path(xp_base_path) / xp_name / graph_name / query_hash / device_id;

    std::string git_sha = get_git_sha();
    std::ostringstream filename;
    filename << timestamp << "_" << git_sha << "_" << variant << ".csv";

    std::filesystem::path full_path = dir_path / filename.str();
    return full_path.string();
}

} // namespace gpusssp::experiments

#endif
