#include "common/constants.hpp"
#include "common/csv.hpp"
#include "common/dijkstra.hpp"
#include "common/files.hpp"
#include "common/id_queue.hpp"
#include "common/logger.hpp"
#include "common/statistics.hpp"
#include "experiment_util.hpp"
#include "queries.hpp"

#include <chrono>
#include <vector>

using namespace gpusssp;

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        common::log_error() << "Usage: " << argv[0] << " <base_path>" << std::endl;
        return 1;
    }

    std::string base_path = argv[1];

    common::log() << "Loading graph from: " << base_path << std::endl;
    auto graph = common::files::read_weighted_graph<uint32_t>(base_path);
    common::log() << "Graph loaded: " << graph.num_nodes() << " nodes." << std::endl;

    common::log() << "Loading queries from: " << base_path << "/queries.csv" << std::endl;
    auto queries = experiments::read_queries(base_path);
    common::log() << "Loaded " << queries.size() << " queries." << std::endl;

    // Generate output filename
    uint64_t timestamp = experiments::get_unix_timestamp();
    std::string queries_hash = experiments::hash_queries_content(queries);
    std::string device_hash = experiments::hash_device_name("cpu");
    std::string output_filename = experiments::generate_experiment_filename(
        timestamp, queries_hash, device_hash, "", "dijkstra");
    common::log() << "Output file: " << output_filename << std::endl;

    common::MinIDQueue queue(graph.num_nodes());
    common::CostVector<common::WeightedGraph<uint32_t>> costs(graph.num_nodes(),
                                                              common::INF_WEIGHT);
    std::vector<bool> settled(graph.num_nodes(), false);

    common::CSVWriter<uint32_t, uint32_t, std::optional<uint8_t>, uint32_t, uint64_t> writer(
        output_filename);
    writer.write_header({"from_node_id", "to_node_id", "rank", "distance", "time"});

    common::log() << "Running queries..." << std::endl;

    int progress_counter = 0;
    uint64_t total_duration = 0;
    for (const auto &query : queries)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        auto dist = common::dijkstra(query.from, query.to, graph, queue, costs, settled);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

        writer.write({query.from, query.to, query.rank, dist, duration});

        total_duration += duration;
        progress_counter++;
        if (progress_counter % 100 == 0)
        {
            common::log() << "Processed " << progress_counter << " queries." << std::endl;
        }
    }

    common::log() << "Done." << std::endl;
    common::log() << "Processed " << queries.size() << " queries in "
                  << (total_duration / queries.size()) << "us/req (average)" << std::endl;

#ifdef ENABLE_STATISTICS
    common::log() << "Statistics: " << std::endl
                  << common::Statistics::get().summary() << std::endl;
#endif

    return 0;
}
