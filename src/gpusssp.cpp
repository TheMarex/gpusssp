#include <cassert>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <random>
#include <vector>
#include <vulkan/vulkan.hpp>

#include "common/coordinate.hpp"
#include "common/dijkstra.hpp"
#include "common/files.hpp"
#include "common/id_queue.hpp"
#include "common/lazy_clear_vector.hpp"
#include "common/nearest_neighbour.hpp"
#include "common/shader.hpp"
#include "common/timed_logger.hpp"
#include "common/weighted_graph.hpp"

#include "gpu/deltastep.hpp"
#include "gpu/deltastep_buffers.hpp"
#include "gpu/graph_buffers.hpp"
#include "gpu/memory.hpp"

using namespace gpusssp;

const uint32_t WORKGROUP_SIZE = 128;

std::optional<common::Coordinate> string_to_coordinate(const std::string &s)
{
    if (s == "random")
    {
        return {};
    }

    auto pos = s.find(',');
    if (pos == s.npos)
    {
        return {};
    }
    return common::Coordinate::from_floating(std::stod(s.substr(0, pos)),
                                             std::stod(s.substr(pos + 1)));
}

std::optional<uint32_t> string_to_node_id(const std::string &s)
{
    if (s.size() < 1 || !std::isdigit(s[0]))
    {
        return {};
    }

    std::size_t pos;
    auto node_id = std::stoi(s, &pos);
    if (pos != s.size())
    {
        return {};
    }

    return node_id;
}

int main(int argc, char **argv)
{
    // Parse command line arguments
    if (argc < 2 || argc > 6)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <graph_path> [SRC_LON,SRC_LAT DEST_LON,DEST_LAT] [DELTA] [NUM_QUERIES]"
                  << std::endl;
        std::cerr << "Example: " << argv[0]
                  << " cache/berlin 13.3889,52.5170 13.4050,52.5200 3600 1" << std::endl;
        return 1;
    }

    std::string graph_path = argv[1];
    std::optional<common::Coordinate> maybe_src_coord;
    std::optional<common::Coordinate> maybe_dst_coord;
    if (argc > 2)
    {
        maybe_src_coord = string_to_coordinate(argv[2]);
        maybe_dst_coord = string_to_coordinate(argv[3]);
    }
    std::optional<uint32_t> maybe_src_node_id;
    std::optional<uint32_t> maybe_dst_node_id;
    if (argc > 2)
    {
        if (!maybe_src_coord)
            maybe_src_node_id = string_to_node_id(argv[2]);
        if (!maybe_dst_coord)
            maybe_dst_node_id = string_to_node_id(argv[3]);
    }

    auto delta = 3600u;
    if (argc >= 5)
    {
        delta = std::stoi(argv[4]);
    }

    auto num_queries = 1u;
    if (argc >= 6)
    {
        num_queries = std::stoi(argv[5]);
    }

    std::cout << "Loading graph from: " << graph_path << std::endl;
    auto graph = common::files::read_weighted_graph<uint32_t>(graph_path);
    auto coordinates = common::files::read_coordinates(graph_path);

    auto num_heavy = 0u;
    for (uint32_t eid = 0u; eid < graph.num_edges(); ++eid)
    {
        num_heavy += graph.weight(eid) >= delta;
    }

    std::cout << "Graph loaded: " << graph.num_nodes() << " nodes, " << graph.num_edges()
              << " edges (" << num_heavy << " heavy)" << std::endl;

    common::NearestNeighbour nn(coordinates);

    std::uniform_int_distribution<> random_node_id(0, graph.num_nodes() - 1);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<uint32_t> src_nodes(num_queries);
    std::vector<uint32_t> dst_nodes(num_queries);
    if (maybe_src_coord)
    {
        std::fill(src_nodes.begin(), src_nodes.end(), nn.nearest(*maybe_src_coord));
    }
    else if (maybe_src_node_id)
    {
        std::fill(src_nodes.begin(), src_nodes.end(), *maybe_src_node_id);
    }
    else
    {
        std::generate(src_nodes.begin(), src_nodes.end(), [&]() { return random_node_id(gen); });
    }
    if (maybe_dst_coord)
    {
        std::fill(dst_nodes.begin(), dst_nodes.end(), nn.nearest(*maybe_dst_coord));
    }
    else if (maybe_dst_node_id)
    {
        std::fill(dst_nodes.begin(), dst_nodes.end(), *maybe_dst_node_id);
    }
    else
    {
        std::generate(dst_nodes.begin(), dst_nodes.end(), [&]() { return random_node_id(gen); });
    }

    vk::ApplicationInfo appInfo("DeltaStep", 1, "NoEngine", 1, VK_API_VERSION_1_2);
    vk::Instance instance = vk::createInstance({{}, &appInfo});
    auto physDevices = instance.enumeratePhysicalDevices();
    
    // Select device based on GPUSSSP_DEVICE environment variable
    uint32_t device_index = 0;
    if (const char* env_device = std::getenv("GPUSSSP_DEVICE"))
    {
        device_index = std::stoi(env_device);
        if (device_index >= physDevices.size())
        {
            std::cerr << "Error: GPUSSSP_DEVICE=" << device_index 
                      << " is out of range. Found " << physDevices.size() 
                      << " device(s)." << std::endl;
            instance.destroy();
            return 1;
        }
    }
    vk::PhysicalDevice phys = physDevices[device_index];
    std::cout << "Using device " << device_index << ": " 
              << phys.getProperties().deviceName << std::endl;

    float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo queueInfo({}, 0, 1, &queuePriority);
    vk::Device device = phys.createDevice({{}, 1, &queueInfo});
    vk::Queue queue = device.getQueue(0, 0);

    vk::CommandPool cmdPool =
        device.createCommandPool({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, 0});

    {
        common::MinIDQueue min_queue(graph.num_nodes());
        common::CostVector<common::WeightedGraph<uint32_t>> costs(graph.num_nodes(),
                                                                  common::INF_WEIGHT);
        std::vector<bool> settled(graph.num_nodes(), false);

        gpu::GraphBuffers graph_buffers(graph, device);
        gpu::DeltaStepBuffers deltastep_buffers(graph.num_nodes(), device);

        graph_buffers.initialize(phys.getMemoryProperties());
        deltastep_buffers.initialize(phys.getMemoryProperties());

        gpu::DeltaStep deltastep(graph_buffers, deltastep_buffers, device);
        deltastep.initialize();

        std::uint32_t checksum = 0;
        std::uint32_t dij_duration = 0;
        std::uint32_t ds_duration = 0;
        for (auto i = 0u; i < num_queries; i++)
        {
            auto time_1 = std::chrono::high_resolution_clock::now();
            auto expected_dist =
                common::dijkstra(src_nodes[i], dst_nodes[i], graph, min_queue, costs, settled);
            auto time_2 = std::chrono::high_resolution_clock::now();
            auto dist = deltastep.run(cmdPool, queue, src_nodes[i], dst_nodes[i], delta);
            auto time_3 = std::chrono::high_resolution_clock::now();

            dij_duration +=
                std::chrono::duration_cast<std::chrono::milliseconds>(time_2 - time_1).count();
            ds_duration +=
                std::chrono::duration_cast<std::chrono::milliseconds>(time_3 - time_2).count();
            if (dist != expected_dist)
            {
                std::cout << "Error: Distance " << src_nodes[i] << "->" << dst_nodes[i]
                          << " mismatch. expected: " << expected_dist << " actual: " << dist
                          << std::endl;

                auto *gpu_dist = deltastep_buffers.dist();
                for (auto node_id = 0u; node_id < graph.num_nodes(); ++node_id)
                {
                    if (gpu_dist[node_id] != common::INF_WEIGHT)
                    {
                        std::cout << "\t" << node_id << "\t" << gpu_dist[node_id];
                        if (gpu_dist[node_id] != costs[node_id])
                        {
                            std::cout << " mismatch " << costs[node_id]
                                      << (settled[node_id] ? " setteled" : " ");
                        }
                        std::cout << std::endl;
                    }
                }
            }
            checksum ^= dist;
        }
        std::cout << "Processed " << num_queries << " queries in " << dij_duration
                  << "ms (dijkstra) " << ds_duration << "ms (deltastep)" << std::endl;
        std::cout << "Checksum: " << checksum << std::endl;
    }

    device.destroyCommandPool(cmdPool);
    device.destroy();
    instance.destroy();

    return 0;
}
