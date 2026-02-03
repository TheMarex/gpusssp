#include "common/csv.hpp"
#include "common/files.hpp"
#include "common/logger.hpp"
#include "common/statistics.hpp"
#include "common/weighted_graph.hpp"
#include "experiment_util.hpp"
#include "queries.hpp"

#include "gpu/bellmanford.hpp"
#include "gpu/bellmanford_buffers.hpp"
#include "gpu/graph_buffers.hpp"
#include "gpu/statistics.hpp"
#include "gpu/vulkan_context.hpp"

#include <chrono>
#include <vector>
#include <vulkan/vulkan.hpp>

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
    gpu::VulkanContext vk_ctx("BellmanFordExperiment", gpu::detail::selectDevice());
    uint64_t timestamp = experiments::get_unix_timestamp();
    std::string queries_hash = experiments::hash_queries_content(queries);
    std::string device_hash = experiments::hash_device_name(vk_ctx.device_name());
    std::string output_filename =
        experiments::generate_experiment_filename(timestamp, queries_hash, device_hash, "", "bellmanford");
    common::log() << "Output file: " << output_filename << std::endl;

    auto device = vk_ctx.device();
    auto queue = vk_ctx.queue();
    auto cmdPool = vk_ctx.command_pool();

    common::CSVWriter<uint32_t, uint32_t, std::optional<uint8_t>, uint32_t, uint64_t> writer(
        output_filename);
    writer.write_header({"from_node_id", "to_node_id", "rank", "distance", "time"});

    common::log() << "Running queries..." << std::endl;

    {
        gpu::GraphBuffers graph_buffers(graph, device, vk_ctx.memory_properties(), cmdPool, queue);
        gpu::BellmanFordBuffers bellmanford_buffers(
            graph.num_nodes(), device, vk_ctx.memory_properties());
        gpu::Statistics gpu_statistics(device, vk_ctx.memory_properties());

        gpu::BellmanFord bellmanford(graph_buffers, bellmanford_buffers, device, gpu_statistics);
        bellmanford.initialize();

        int progress_counter = 0;
        uint64_t total_duration = 0;
        for (const auto &query : queries)
        {
            auto start_time = std::chrono::high_resolution_clock::now();

            auto dist = bellmanford.run(cmdPool, queue, query.from, query.to);

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)
                    .count();

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
                      << common::Statistics::get().summary() << gpu_statistics.summary() << std::endl;
#endif
    }

    return 0;
}
