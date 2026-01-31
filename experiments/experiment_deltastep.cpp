#include "common/csv.hpp"
#include "common/files.hpp"
#include "common/weighted_graph.hpp"
#include "experiment_util.hpp"
#include "queries.hpp"

#include "gpu/deltastep.hpp"
#include "gpu/deltastep_buffers.hpp"
#include "gpu/graph_buffers.hpp"
#include "gpu/vulkan_context.hpp"

#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>
#include <vulkan/vulkan.hpp>

using namespace gpusssp;

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3)
    {
        std::cerr << "Usage: " << argv[0] << " <base_path> [delta]" << std::endl;
        return 1;
    }

    std::string base_path = argv[1];

    // Default delta value (3600 seconds = 1 hour, typical for road networks)
    uint32_t delta = 3600u;
    if (argc >= 3)
    {
        delta = std::stoi(argv[2]);
    }

    std::cout << "Loading graph from: " << base_path << std::endl;
    auto graph = common::files::read_weighted_graph<uint32_t>(base_path);
    std::cout << "Graph loaded: " << graph.num_nodes() << " nodes." << std::endl;

    std::cout << "Loading queries from: " << base_path << "/queries.csv" << std::endl;
    auto queries = experiments::read_queries(base_path);
    std::cout << "Loaded " << queries.size() << " queries." << std::endl;

    // Generate output filename with delta parameter
    gpu::VulkanContext vk_ctx("DeltaStepExperiment", gpu::detail::selectDevice());
    uint64_t timestamp = experiments::get_unix_timestamp();
    std::string queries_hash = experiments::hash_queries_content(queries);
    std::string device_hash = experiments::hash_device_name(vk_ctx.device_name());
    std::ostringstream params_stream;
    params_stream << "delta" << delta;
    std::string output_filename = experiments::generate_experiment_filename(
        timestamp, queries_hash, device_hash, params_stream.str(), "deltastep");
    std::cout << "Output file: " << output_filename << std::endl;

    auto device = vk_ctx.device();
    auto queue = vk_ctx.queue();
    auto cmdPool = vk_ctx.command_pool();

    common::CSVWriter<uint32_t, uint32_t, std::optional<uint8_t>, uint32_t, uint64_t> writer(
        output_filename);
    writer.write_header({"from_node_id", "to_node_id", "rank", "distance", "time"});

    std::cout << "Running queries with delta = " << delta << "..." << std::endl;

    {
        gpu::GraphBuffers graph_buffers(graph, device, vk_ctx.memory_properties(), cmdPool, queue);
        gpu::DeltaStepBuffers deltastep_buffers(
            graph.num_nodes(), device, vk_ctx.memory_properties());

        gpu::DeltaStep deltastep(graph_buffers, deltastep_buffers, device);
        deltastep.initialize();

        int progress_counter = 0;
        for (const auto &query : queries)
        {
            auto start_time = std::chrono::high_resolution_clock::now();

            auto dist = deltastep.run(cmdPool, queue, query.from, query.to, delta);

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)
                    .count();

            writer.write({query.from, query.to, query.rank, dist, duration});

            progress_counter++;
            if (progress_counter % 100 == 0)
            {
                std::cout << "Processed " << progress_counter << " queries." << std::endl;
            }
        }
    }

    std::cout << "Done." << std::endl;

    return 0;
}
