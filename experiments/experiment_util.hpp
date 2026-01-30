#ifndef GPUSSSP_EXPERIMENTS_EXPERIMENT_UTIL_HPP
#define GPUSSSP_EXPERIMENTS_EXPERIMENT_UTIL_HPP

#include "queries.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace gpusssp::experiments
{

inline uint64_t get_unix_timestamp()
{
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::seconds>(duration).count();
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

inline std::string generate_experiment_filename(uint64_t timestamp,
                                                const std::string &queries_hash,
                                                const std::string &params,
                                                const std::string &algorithm)
{
    std::ostringstream filename;
    filename << timestamp << "_" << queries_hash << "_";

    if (!params.empty())
    {
        filename << params << "_";
    }

    filename << algorithm << ".csv";

    return filename.str();
}

} // namespace gpusssp::experiments

#endif
