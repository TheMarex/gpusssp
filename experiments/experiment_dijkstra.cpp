#include "common/constants.hpp"
#include "common/csv.hpp"
#include "common/dijkstra.hpp"
#include "common/files.hpp"
#include "common/id_queue.hpp"
#include "common/logger.hpp"
#include "common/progress_bar.hpp"
#include "common/statistics.hpp"
#include "experiment_util.hpp"
#include "queries.hpp"

#include <chrono>
#include <vector>

using namespace gpusssp;

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        common::log_error() << "Usage: " << argv[0] << " <graph_base_path> <xp_base_path> [xp_name]"
                            << std::endl;
        return 1;
    }

    std::string graph_base_path = argv[1];
    std::string xp_base_path = argv[2];

    std::string xp_name = "compare_algorithm";
    if (argc >= 4)
    {
        xp_name = argv[3];
    }

    common::log() << "Loading graph from: " << graph_base_path << std::endl;
    auto graph = common::files::read_weighted_graph<uint32_t>(graph_base_path);
    common::log() << "Graph loaded: " << graph.num_nodes() << " nodes." << std::endl;

    common::log() << "Loading queries from: " << graph_base_path << "/queries.csv" << std::endl;
    auto queries = experiments::read_queries(graph_base_path);
    common::log() << "Loaded " << queries.size() << " queries." << std::endl;

    uint64_t timestamp = experiments::get_unix_timestamp();
    std::string query_hash = experiments::hash_queries_content(queries);
    std::string device_id = "cpu";
    std::string graph_name = experiments::extract_graph_name(graph_base_path);
    std::string variant = "dijkstra";

    experiments::create_experiment_directories(
        xp_base_path, xp_name, graph_name, query_hash, device_id);
    std::string output_filename = experiments::generate_experiment_filename(
        xp_base_path, xp_name, graph_name, query_hash, device_id, timestamp, variant);
    common::log() << "Output file: " << output_filename << std::endl;

    common::MinIDQueue queue(graph.num_nodes());
    common::CostVector<common::WeightedGraph<uint32_t>> costs(graph.num_nodes(),
                                                              common::INF_WEIGHT);
    std::vector<bool> settled(graph.num_nodes(), false);

    common::CSVWriter<uint32_t, uint32_t, std::optional<uint8_t>, uint32_t, uint64_t> writer(
        output_filename);
    writer.write_header({"from_node_id", "to_node_id", "rank", "distance", "time"});

    common::log() << "Running queries..." << std::endl;

    common::ProgressBar progress_bar(queries.size());
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
        progress_bar.increment();
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
