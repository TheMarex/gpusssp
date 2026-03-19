#include "common/bucket_queue.hpp"
#include "common/constants.hpp"
#include "common/csv.hpp"
#include "common/dial.hpp"
#include "common/dijkstra.hpp"
#include "common/files.hpp"
#include "common/logger.hpp"
#include "common/progress_bar.hpp"
#include "common/statistics.hpp"
#include "common/weighted_graph.hpp"
#include "experiment_util.hpp"
#include "queries.hpp"

#include <argparse/argparse.hpp>
#include <chrono>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

using namespace gpusssp;

int main(int argc, char **argv)
{
    argparse::ArgumentParser program("experiment_dial", "1.0.0");
    program.add_description("Run Dial algorithm benchmark.");

    program.add_argument("graph_path").help("path to preprocessed graph data (without extension)");

    program.add_argument("xp_path").help("base path for experiment output");

    program.add_argument("-n", "--name")
        .default_value(std::string("compare_algorithm"))
        .help("experiment name");

    program.add_argument("--range").default_value(32768u).scan<'u', uint32_t>().help(
        "range parameter for bucket queue");

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception &err)
    {
        std::cerr << err.what() << '\n';
        std::cerr << program;
        return 1;
    }

    std::string graph_base_path = program.get("graph_path");
    std::string xp_base_path = program.get("xp_path");
    std::string xp_name = program.get("--name");
    uint32_t range = program.get<uint32_t>("--range");

    common::log() << "Loading graph from: " << graph_base_path << '\n';
    auto graph = common::files::read_weighted_graph<uint32_t>(graph_base_path);
    common::log() << "Graph loaded: " << graph.num_nodes() << " nodes." << '\n';

    common::log() << "Loading queries from: " << graph_base_path << "/queries.csv" << '\n';
    auto queries = experiments::read_queries(graph_base_path);
    common::log() << "Loaded " << queries.size() << " queries." << '\n';

    uint64_t timestamp = experiments::get_unix_timestamp();
    std::string query_hash = experiments::hash_queries_content(queries);
    std::string device_id = "cpu";
    std::string graph_name = experiments::extract_graph_name(graph_base_path);
    std::string variant = "dial";

    experiments::create_experiment_directories(
        xp_base_path, xp_name, graph_name, query_hash, device_id);
    std::string output_filename = experiments::generate_experiment_filename(
        xp_base_path, xp_name, graph_name, query_hash, device_id, timestamp, variant);
    common::log() << "Output file: " << output_filename << '\n';

    common::BucketQueue queue(graph.num_nodes(), range);
    common::CostVector<common::WeightedGraph<uint32_t>> costs(graph.num_nodes(),
                                                              common::INF_WEIGHT);
    std::vector<bool> settled(graph.num_nodes(), false);

    common::CSVWriter<uint32_t, uint32_t, std::optional<uint8_t>, uint32_t, uint64_t> writer(
        output_filename);
    writer.write_header({"from_node_id", "to_node_id", "rank", "distance", "time"});

    common::log() << "Running queries..." << '\n';

    common::ProgressBar progress_bar(queries.size());
    uint64_t total_duration = 0;
    for (const auto &query : queries)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        auto dist = common::dial(query.from, query.to, graph, queue, costs, settled);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

        writer.write({query.from, query.to, query.rank, dist, duration});

        total_duration += duration;
        progress_bar.increment();
    }

    common::log() << "Done." << '\n';
    common::log() << "Processed " << queries.size() << " queries in "
                  << (total_duration / queries.size()) << "us/req (average)" << '\n';

#ifdef ENABLE_STATISTICS
    common::log() << "Statistics: " << std::endl
                  << common::Statistics::get().summary() << std::endl;
#endif

    return 0;
}
