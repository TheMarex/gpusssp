#include "common/dijkstra.hpp"
#include "common/files.hpp"
#include "common/logger.hpp"
#include "queries.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace gpusssp;

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        common::log_error() << "Usage: " << argv[0] << " <mode> <num_queries> <base_path>"
                            << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    uint32_t num_queries = 0;
    try
    {
        num_queries = std::stoi(argv[2]);
    }
    catch (const std::exception &e)
    {
        common::log_error() << "Invalid number of queries: " << argv[2] << std::endl;
        return 1;
    }
    std::string base_path = argv[3];

    if (mode != "random" && mode != "rank")
    {
        common::log_error() << "Supported modes: 'random', 'rank'" << std::endl;
        return 1;
    }

    common::log() << "Loading graph from: " << base_path << std::endl;
    auto graph = common::files::read_weighted_graph<uint32_t>(base_path);

    common::log() << "Graph loaded: " << graph.num_nodes() << " nodes." << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(0, graph.num_nodes() - 1);

    std::vector<experiments::Query> queries;
    queries.reserve(num_queries);

    if (mode == "random")
    {
        for (uint32_t i = 0; i < num_queries; ++i)
        {
            queries.push_back({dist(gen), dist(gen), -1});
        }
        common::log() << "Generated " << queries.size() << " random queries." << std::endl;
    }
    else if (mode == "rank")
    {
        uint32_t num_targets = static_cast<uint32_t>(std::log2(graph.num_nodes()));
        if (num_queries < num_targets)
        {
            common::log() << "Minimum amount of queries for rank is " << num_targets << std::endl;
            return EXIT_FAILURE;
        }
        else if (num_queries % num_targets != 0)
        {
            common::log() << "Number of queries needs to be a multiple of " << num_targets
                          << std::endl;
            return EXIT_FAILURE;
        }
        uint32_t num_sources = static_cast<uint32_t>(num_queries / num_targets);

        common::log() << "Generating rank queries with " << num_sources << " sources and "
                      << num_targets << " targets per source." << std::endl;

        std::vector<uint32_t> sources;
        sources.reserve(num_sources);
        for (uint32_t i = 0; i < num_sources; ++i)
        {
            sources.push_back(dist(gen));
        }

        common::MinIDQueue queue(graph.num_nodes());
        common::CostVector<decltype(graph)> costs(graph.num_nodes(), common::INF_WEIGHT);
        std::vector<uint32_t> nodes(graph.num_nodes());
        std::iota(nodes.begin(), nodes.end(), 0);

        for (uint32_t source : sources)
        {
            std::vector<uint32_t> targets;
            targets.reserve(num_targets);
            for (uint32_t i = 0; i < num_targets; ++i)
            {
                targets.push_back(dist(gen));
            }

            common::dijkstra_to_all(source, graph, queue, costs);

            std::sort(nodes.begin(),
                      nodes.end(),
                      [&](const auto lhs, const auto rhs) { return costs[lhs] < costs[rhs]; });

            for (int rank = 0; rank < num_targets; ++rank)
            {
                auto index = (1u << rank) - 1;
                queries.push_back({source, nodes[index], rank});
            }
        }

        common::log() << "Generated " << queries.size() << " rank queries." << std::endl;
        std::shuffle(queries.begin(), queries.end(), gen);
        common::log() << "Shuffled all rank queries." << std::endl;
    }

    experiments::write_queries(base_path, queries);
    common::log() << "Queries written to " << base_path << "/queries.csv" << std::endl;

    return 0;
}
