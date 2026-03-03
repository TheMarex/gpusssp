#include "common/csv.hpp"
#include "common/files.hpp"
#include "common/logger.hpp"
#include "common/progress_bar.hpp"
#include "common/weighted_graph.hpp"
#include "experiment_util.hpp"
#include "queries.hpp"

#include "gpu/bellmanford.hpp"
#include "gpu/bellmanford_buffers.hpp"
#include "gpu/graph_buffers.hpp"
#include "gpu/statistics.hpp"
#include "gpu/vulkan_context.hpp"

#include <chrono>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <vector>
#include <vulkan/vulkan.hpp>

using namespace gpusssp;

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        common::log_error() << "Usage: " << argv[0] << " <graph_base_path> <xp_base_path> [xp_name]"
                            << '\n';
        return 1;
    }

    std::string graph_base_path = argv[1];
    std::string xp_base_path = argv[2];

    std::string xp_name = "compare_algorithm";
    if (argc >= 4)
    {
        xp_name = argv[3];
    }

    common::log() << "Loading graph from: " << graph_base_path << '\n';
    auto graph = common::files::read_weighted_graph<uint32_t>(graph_base_path);
    common::log() << "Graph loaded: " << graph.num_nodes() << " nodes." << '\n';

    common::log() << "Loading queries from: " << graph_base_path << "/queries.csv" << '\n';
    auto queries = experiments::read_queries(graph_base_path);
    common::log() << "Loaded " << queries.size() << " queries." << '\n';

    gpu::VulkanContext vk_ctx("BellmanFordExperiment", gpu::detail::select_device());
    uint64_t timestamp = experiments::get_unix_timestamp();
    std::string query_hash = experiments::hash_queries_content(queries);
    std::string device_id = experiments::hash_device_name(vk_ctx.device_name());
    std::string graph_name = experiments::extract_graph_name(graph_base_path);
    std::string variant = "bellmanford";

    experiments::create_experiment_directories(
        xp_base_path, xp_name, graph_name, query_hash, device_id);
    std::string output_filename = experiments::generate_experiment_filename(
        xp_base_path, xp_name, graph_name, query_hash, device_id, timestamp, variant);
    common::log() << "Output file: " << output_filename << '\n';

    auto device = vk_ctx.device();
    auto queue = vk_ctx.queue();
    auto cmd_pool = vk_ctx.command_pool();

    common::CSVWriter<uint32_t, uint32_t, std::optional<uint8_t>, uint32_t, uint64_t> writer(
        output_filename);
    writer.write_header({"from_node_id", "to_node_id", "rank", "distance", "time"});

    common::log() << "Running queries..." << '\n';

    {
        gpu::GraphBuffers graph_buffers(graph, device, vk_ctx.memory_properties(), cmd_pool, queue);
        gpu::BellmanFordBuffers bellmanford_buffers(
            graph.num_nodes(), device, vk_ctx.memory_properties());
        gpu::Statistics gpu_statistics(device, vk_ctx.memory_properties());

        gpu::BellmanFord bellmanford(graph_buffers, bellmanford_buffers, device, gpu_statistics);
        bellmanford.initialize(cmd_pool);

        common::ProgressBar progress_bar(queries.size());
        uint64_t total_duration = 0;
        for (const auto &query : queries)
        {
            auto start_time = std::chrono::high_resolution_clock::now();

            auto dist = bellmanford.run(cmd_pool, queue, query.from, query.to);

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)
                    .count();

            writer.write({query.from, query.to, query.rank, dist, duration});

            total_duration += duration;
            progress_bar.increment();
        }

        common::log() << "Done." << '\n';
        common::log() << "Processed " << queries.size() << " queries in "
                      << (total_duration / queries.size()) << "us/req (average)" << '\n';

#ifdef ENABLE_STATISTICS
        common::log() << "Statistics: " << std::endl
                      << common::Statistics::get().summary() << gpu_statistics.summary()
                      << std::endl;
#endif
    }

    return 0;
}
