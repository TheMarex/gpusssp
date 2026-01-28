#include "common/constants.hpp"
#include "common/csv.hpp"
#include "common/files.hpp"
#include "common/shader.hpp"
#include "common/weighted_graph.hpp"
#include "experiment_util.hpp"
#include "queries.hpp"

#include "gpu/deltastep.hpp"
#include "gpu/deltastep_buffers.hpp"
#include "gpu/graph_buffers.hpp"
#include "gpu/memory.hpp"

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

    std::cout << "Loading queries from: " << base_path << "/queries" << std::endl;
    auto queries = experiments::read_queries(base_path);
    std::cout << "Loaded " << queries.size() << " queries." << std::endl;

    // Generate output filename with delta parameter
    uint64_t timestamp = experiments::get_unix_timestamp();
    std::string queries_hash = experiments::hash_queries_content(queries);
    std::ostringstream params_stream;
    params_stream << "delta" << delta;
    std::string output_filename = experiments::generate_experiment_filename(
        timestamp, queries_hash, params_stream.str(), "deltastep");
    std::cout << "Output file: " << output_filename << std::endl;

    // Initialize Vulkan
    vk::ApplicationInfo appInfo("DeltaStepExperiment", 1, "NoEngine", 1, VK_API_VERSION_1_2);
    vk::Instance instance = vk::createInstance({{}, &appInfo});
    auto physDevices = instance.enumeratePhysicalDevices();
    vk::PhysicalDevice phys = physDevices[0];

    float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo queueInfo({}, 0, 1, &queuePriority);
    vk::Device device = phys.createDevice({{}, 1, &queueInfo});
    vk::Queue queue = device.getQueue(0, 0);

    vk::CommandPool cmdPool =
        device.createCommandPool({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, 0});

    common::CSVWriter<uint32_t, uint32_t, uint32_t, long long> writer(output_filename);
    writer.write_header({"from_node_id", "to_node_id", "distance", "time"});

    std::cout << "Running queries with delta = " << delta << "..." << std::endl;

    {
        gpu::GraphBuffers graph_buffers(graph, device);
        gpu::DeltaStepBuffers deltastep_buffers(graph.num_nodes(), device);

        graph_buffers.initialize(phys.getMemoryProperties());
        deltastep_buffers.initialize(phys.getMemoryProperties());

        gpu::DeltaStep deltastep(graph_buffers, deltastep_buffers, device);
        deltastep.initialize();

        int progress_counter = 0;
        for (const auto &query : queries)
        {
            auto start_time = std::chrono::high_resolution_clock::now();

            auto dist = deltastep.run(cmdPool, queue, query.from, query.to, delta);

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

            writer.write({query.from, query.to, dist, duration});

            progress_counter++;
            if (progress_counter % 100 == 0)
            {
                std::cout << "Processed " << progress_counter << " queries." << std::endl;
            }
        }
    }

    device.destroyCommandPool(cmdPool);
    device.destroy();
    instance.destroy();

    std::cout << "Done." << std::endl;

    return 0;
}
