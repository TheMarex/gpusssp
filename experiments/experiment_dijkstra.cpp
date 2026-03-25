#include "experiment_util.hpp"
#include "queries.hpp"
#include "common/constants.hpp"
#include "common/csv.hpp"
#include "common/dijkstra.hpp"
#include "common/files.hpp"
#include "common/id_queue.hpp"
#include "common/logger.hpp"
#include "common/progress_bar.hpp"
#include "common/statistics.hpp"
#include "common/weighted_graph.hpp"

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
    argparse::ArgumentParser program("experiment_dijkstra", "1.0.0");
    program.add_description("Run Dijkstra algorithm benchmark.");

    program.add_argument("graph_path").help("path to preprocessed graph data (without extension)");

    program.add_argument("xp_path").help("base path for experiment output");

    program.add_argument("-n", "--name")
        .default_value(std::string("compare_algorithm"))
        .help("experiment name");

    program.add_argument("--metrics")
        .default_value(std::string("time"))
        .help("comma-separated metrics to capture: time, edges_relaxed");

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception &err)
    {
        common::log_error() << err.what() << '\n' << program;
        return 1;
    }

    std::string graph_base_path = program.get("graph_path");
    std::string xp_base_path = program.get("xp_path");
    std::string xp_name = program.get("--name");

    auto metrics = experiments::parse_metrics(program.get("--metrics"));
    experiments::validate_metrics(metrics);

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
    std::string variant = "dijkstra";

    experiments::create_experiment_directories(
        xp_base_path, xp_name, graph_name, query_hash, device_id);
    std::string output_filename = experiments::generate_experiment_filename(
        xp_base_path, xp_name, graph_name, query_hash, device_id, timestamp, variant);
    common::log() << "Output file: " << output_filename << '\n';

    common::MinIDQueue queue(graph.num_nodes());
    common::CostVector<common::WeightedGraph<uint32_t>> costs(graph.num_nodes(),
                                                              common::INF_WEIGHT);
    std::vector<bool> settled(graph.num_nodes(), false);

    std::vector<std::string> headers = {"from_node_id", "to_node_id", "rank", "distance"};
    for (const auto &metric : metrics)
    {
        headers.push_back(metric);
    }

    common::CSVWriter<uint32_t, uint32_t, std::optional<uint8_t>, uint32_t, std::vector<uint64_t>>
        writer(output_filename);
    writer.write_header(headers);

    common::log() << "Running queries..." << '\n';

    common::ProgressBar progress_bar(queries.size());
    std::vector<uint64_t> metric_values;
    metric_values.reserve(metrics.size());

    for (const auto &query : queries)
    {
        metric_values.clear();

        auto start_time = std::chrono::high_resolution_clock::now();
        auto start_edges = common::Statistics::get().value(common::StatisticsEvent::DIJKSTRA_RELAX);

        auto dist = common::dijkstra(query.from, query.to, graph, queue, costs, settled);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        auto edges_relaxed =
            common::Statistics::get().value(common::StatisticsEvent::DIJKSTRA_RELAX) - start_edges;

        for (const auto &metric : metrics)
        {
            if (metric == "time")
            {
                metric_values.push_back(duration);
            }
            else if (metric == "edges_relaxed")
            {
                metric_values.push_back(edges_relaxed);
            }
        }

        writer.write({query.from, query.to, query.rank, dist, metric_values});

        progress_bar.increment();
    }

    common::log() << "Done." << '\n';

#ifdef ENABLE_STATISTICS
    common::log() << "Statistics: " << '\n' << common::Statistics::get().summary() << '\n';
#endif

    return 0;
}
