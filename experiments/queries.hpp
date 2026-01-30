#ifndef GPUSSSP_EXPERIMENTS_QUERIES_HPP
#define GPUSSSP_EXPERIMENTS_QUERIES_HPP

#include "common/csv.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace gpusssp::experiments
{
struct Query
{
    uint32_t from;
    uint32_t to;
    std::optional<uint8_t> rank;
};

inline auto read_queries(const std::string &base_path)
{
    std::vector<Query> queries;
    std::string queries_file = base_path + "/queries.csv";

    common::CSVReader<uint32_t, uint32_t, std::optional<uint8_t>> reader(queries_file);
    std::vector<std::string> header;
    reader.read_header(header);

    std::tuple<uint32_t, uint32_t, std::optional<uint8_t>> row;
    while (reader.read(row))
    {
        queries.push_back({std::get<0>(row), std::get<1>(row), std::get<2>(row)});
    }

    return queries;
}

inline void write_queries(const std::string &base_path, const std::vector<Query> &queries)
{
    std::string queries_file = base_path + "/queries.csv";

    // Check if any query has a rank
    bool has_rank = false;
    for (const auto &query : queries)
    {
        if (query.rank)
        {
            has_rank = true;
            break;
        }
    }

    if (has_rank)
    {
        common::CSVWriter<uint32_t, uint32_t, int32_t> writer(queries_file);
        writer.write_header({"from_node_id", "to_node_id", "rank"});
        for (const auto &query : queries)
        {
            writer.write({query.from, query.to, *query.rank});
        }
    }
    else
    {
        common::CSVWriter<uint32_t, uint32_t> writer(queries_file);
        writer.write_header({"from_node_id", "to_node_id"});
        for (const auto &query : queries)
        {
            writer.write({query.from, query.to});
        }
    }
}
} // namespace gpusssp::experiments

#endif
