#include "common/constants.hpp"
#include "common/csv.hpp"
#include "common/dijkstra.hpp"
#include "common/files.hpp"
#include "common/id_queue.hpp"
#include "common/lazy_clear_vector.hpp"
#include "experiment_util.hpp"
#include "queries.hpp"

#include <chrono>
#include <iostream>
#include <vector>

using namespace gpusssp;

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <base_path>" << std::endl;
        return 1;
    }

    std::string base_path = argv[1];

    std::cout << "Loading graph from: " << base_path << std::endl;
    auto graph = common::files::read_weighted_graph<uint32_t>(base_path);
    std::cout << "Graph loaded: " << graph.num_nodes() << " nodes." << std::endl;

    std::cout << "Loading queries from: " << base_path << "/queries.csv" << std::endl;
    auto queries = experiments::read_queries(base_path);
    std::cout << "Loaded " << queries.size() << " queries." << std::endl;

    // Generate output filename
    uint64_t timestamp = experiments::get_unix_timestamp();
    std::string queries_hash = experiments::hash_queries_content(queries);
    std::string output_filename = experiments::generate_experiment_filename(
        timestamp, queries_hash, "", "dijkstra");
    std::cout << "Output file: " << output_filename << std::endl;

    common::MinIDQueue queue(graph.num_nodes());
    common::CostVector<common::WeightedGraph<uint32_t>> costs(graph.num_nodes(), common::INF_WEIGHT);
    std::vector<bool> settled(graph.num_nodes(), false);

    common::CSVWriter<uint32_t, uint32_t, uint32_t, long long> writer(output_filename);
    writer.write_header({"from_node_id", "to_node_id", "distance", "time"});

    std::cout << "Running queries..." << std::endl;

    int progress_counter = 0;
    for (const auto &query : queries)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        auto dist = common::dijkstra(query.from, query.to, graph, queue, costs, settled);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

        writer.write({query.from, query.to, dist, duration});

        progress_counter++;
        if (progress_counter % 100 == 0)
        {
            std::cout << "Processed " << progress_counter << " queries." << std::endl;
        }
    }

    std::cout << "Done." << std::endl;

    return 0;
}
