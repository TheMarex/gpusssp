#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <optional>
#include <random>
#include <vulkan/vulkan.hpp>

#include "common/files.hpp"
#include "common/shader.hpp"
#include "common/timed_logger.hpp"
#include "common/coordinate.hpp"
#include "common/weighted_graph.hpp"
#include "common/nearest_neighbour.hpp"

#include "gpu/memory.hpp"
#include "gpu/graph_buffers.hpp"
#include "gpu/deltastep_buffers.hpp"
#include "gpu/deltastep.hpp"

using namespace gpusssp;

const uint32_t WORKGROUP_SIZE = 128;

std::optional<common::Coordinate> string_to_coordinate(const std::string& s) {
    if (s == "random") {
      return {};
    }

    auto pos = s.find(',');
    if (pos == s.npos) {
      throw new std::runtime_error("Illegal coordinate " + s);
    }
    return common::Coordinate::from_floating(std::stod(s.substr(0, pos)), std::stod(s.substr(pos+1)));
}


int main(int argc, char **argv)
{
    // Parse command line arguments
    if (argc < 2 || argc > 6) {
        std::cerr << "Usage: " << argv[0] 
                  << " <graph_path> [SRC_LON,SRC_LAT DEST_LON,DEST_LAT] [DELTA] [NUM_QUERIES]" << std::endl;
        std::cerr << "Example: " << argv[0] 
                  << " cache/berlin 13.3889,52.5170 13.4050,52.5200 3600 10" << std::endl;
        return 1;
    }
    
    std::string graph_path = argv[1];
    std::optional<common::Coordinate> maybe_src_coord;
    std::optional<common::Coordinate> maybe_dst_coord;
    if (argc > 2) {
        maybe_src_coord = string_to_coordinate(argv[2]);
        maybe_dst_coord = string_to_coordinate(argv[3]);
    }

    auto delta = 3600u;
    if (argc >= 5) {
      delta = std::stoi(argv[4]);
    }

    auto num_queries = 100u;
    if (argc >= 6) {
      num_queries = std::stoi(argv[5]);
    }
    
    std::cout << "Loading graph from: " << graph_path << std::endl;
    auto graph = common::files::read_weighted_graph<uint32_t>(graph_path);
    auto coordinates = common::files::read_coordinates(graph_path);
    std::cout << "Graph loaded: " << graph.num_nodes() << " nodes, " 
              << graph.num_edges() << " edges" << std::endl;
    
    common::NearestNeighbour nn(coordinates);

    std::uniform_int_distribution<> random_node_id(0, graph.num_nodes()-1);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<uint32_t> src_nodes(num_queries);
    std::vector<uint32_t> dst_nodes(num_queries);
    if (maybe_src_coord) {
        std::fill(src_nodes.begin(), src_nodes.end(), nn.nearest(*maybe_src_coord));
    } else {
        std::generate(src_nodes.begin(), src_nodes.end(), [&]() { return random_node_id(gen); });
    }
    if (maybe_dst_coord) {
        std::fill(dst_nodes.begin(), dst_nodes.end(), nn.nearest(*maybe_dst_coord));
    } else {
        std::generate(dst_nodes.begin(), dst_nodes.end(), [&]() { return random_node_id(gen); });
    }

    vk::ApplicationInfo appInfo("DeltaStep", 1, "NoEngine", 1, VK_API_VERSION_1_2);
    vk::Instance instance = vk::createInstance({{}, &appInfo});
    auto physDevices = instance.enumeratePhysicalDevices();
    vk::PhysicalDevice phys = physDevices[0];

    float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo queueInfo({}, 0, 1, &queuePriority);
    vk::Device device = phys.createDevice({{}, 1, &queueInfo});
    vk::Queue queue = device.getQueue(0, 0);

    vk::CommandPool cmdPool =
        device.createCommandPool({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, 0});
    
    gpu::GraphBuffers graph_buffers(graph, device);
    gpu::DeltaStepBuffers deltastep_buffers(graph.num_nodes(), device);

    graph_buffers.initialize(phys.getMemoryProperties());
    deltastep_buffers.initialize(phys.getMemoryProperties());

    gpu::DeltaStep deltastep(graph_buffers, deltastep_buffers, device);
    deltastep.initialize();

    common::TimedLogger time_query("Running query with delta " + std::to_string(delta));
    std::uint32_t checksum = 0;
    for (int i = 0; i < 10; i++) {
        checksum ^= deltastep.run(cmdPool, queue, src_nodes[i], dst_nodes[i], delta);
    }
    time_query.finished();
    std::cout << "Checksum: " << checksum << std::endl;

    device.destroyCommandPool(cmdPool);
    device.destroy();
    instance.destroy();

    return 0;
}
