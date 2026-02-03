#include "common/csv.hpp"
#include "common/files.hpp"
#include "common/logger.hpp"
#include "common/statistics.hpp"
#include "common/weighted_graph.hpp"
#include "experiment_util.hpp"
#include "queries.hpp"

#include "gpu/graph_buffers.hpp"
#include "gpu/nearfar.hpp"
#include "gpu/nearfar_buffers.hpp"
#include "gpu/vulkan_context.hpp"

#include <chrono>
#include <sstream>
#include <vector>
#include <vulkan/vulkan.hpp>

using namespace gpusssp;

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3)
    {
        common::log_error() << "Usage: " << argv[0] << " <base_path> [delta]" << std::endl;
        return 1;
    }

    std::string base_path = argv[1];

    uint32_t delta = 3600u;
    if (argc >= 3)
    {
        delta = std::stoi(argv[2]);
    }

    common::log() << "Loading graph from: " << base_path << std::endl;
    auto graph = common::files::read_weighted_graph<uint32_t>(base_path);
    common::log() << "Graph loaded: " << graph.num_nodes() << " nodes." << std::endl;

    common::log() << "Loading queries from: " << base_path << "/queries.csv" << std::endl;
    auto queries = experiments::read_queries(base_path);
    common::log() << "Loaded " << queries.size() << " queries." << std::endl;

    gpu::VulkanContext vk_ctx("NearFarExperiment", gpu::detail::selectDevice());
    uint64_t timestamp = experiments::get_unix_timestamp();
    std::string queries_hash = experiments::hash_queries_content(queries);
    std::string device_hash = experiments::hash_device_name(vk_ctx.device_name());
    std::ostringstream params_stream;
    params_stream << "delta" << delta;
    std::string output_filename = experiments::generate_experiment_filename(
        timestamp, queries_hash, device_hash, params_stream.str(), "nearfar");
    common::log() << "Output file: " << output_filename << std::endl;

    auto device = vk_ctx.device();
    auto queue = vk_ctx.queue();
    auto cmdPool = vk_ctx.command_pool();

    common::CSVWriter<uint32_t, uint32_t, std::optional<uint8_t>, uint32_t, uint64_t> writer(
        output_filename);
    writer.write_header({"from_node_id", "to_node_id", "rank", "distance", "time"});

    common::log() << "Running queries with delta = " << delta << "..." << std::endl;

    {
        gpu::GraphBuffers graph_buffers(graph, device, vk_ctx.memory_properties(), cmdPool, queue);
        gpu::NearFarBuffers nearfar_buffers(graph.num_nodes(), device, vk_ctx.memory_properties());
        gpu::Statistics statistics(device, vk_ctx.memory_properties());
        gpu::NearFar nearfar(graph_buffers, nearfar_buffers, device, statistics);
        nearfar.initialize();

        int progress_counter = 0;
        uint64_t total_duration = 0;
        for (const auto &query : queries)
        {
            auto start_time = std::chrono::high_resolution_clock::now();

            auto dist = nearfar.run(cmdPool, queue, query.from, query.to, delta);

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
                      << common::Statistics::get().summary() << statistics.summary() << std::endl;
#endif
    }

    return 0;
}
