#include "common/files.hpp"
#include "queries.hpp"

#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace gpusssp;

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <mode> <num_queries> <base_path>" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    uint32_t num_queries = 0;
    try {
        num_queries = std::stoi(argv[2]);
    } catch (const std::exception& e) {
        std::cerr << "Invalid number of queries: " << argv[2] << std::endl;
        return 1;
    }
    std::string base_path = argv[3];

    if (mode != "random")
    {
        std::cerr << "Only 'random' mode is currently supported." << std::endl;
        return 1;
    }

    std::cout << "Loading graph from: " << base_path << std::endl;
    auto graph = common::files::read_weighted_graph<uint32_t>(base_path);
    
    std::cout << "Graph loaded: " << graph.num_nodes() << " nodes." << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(0, graph.num_nodes() - 1);

    std::vector<experiments::Query> queries;
    queries.reserve(num_queries);

    for (uint32_t i = 0; i < num_queries; ++i)
    {
        queries.push_back({dist(gen), dist(gen)});
    }

    std::cout << "Generated " << queries.size() << " random queries." << std::endl;

    experiments::write_queries(base_path, queries);
    std::cout << "Queries written to " << base_path << "/queries" << std::endl;

    return 0;
}
