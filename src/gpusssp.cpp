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


auto run_gpu_sssp(uint32_t src_node, uint32_t dst_node, uint32_t num_nodes, uint32_t delta, uint32_t* gpuDist, uint32_t* gpuChanged, vk::CommandBuffer& cmdBuf, vk::Queue& queue, vk::Pipeline& pipeline, vk::PipelineLayout& pipelineLayout, vk::DescriptorSet& descSet) {
    struct PushConsts
    {
        uint32_t src_node;
        uint32_t n;
        uint32_t bucket_idx;
        uint32_t delta;
        uint32_t iteration;
    };
    const std::size_t MAX_BUCKETS = UINT32_MAX / delta - 1;


    for (uint32_t bucket = 0; bucket < MAX_BUCKETS; bucket++)
    {
        *gpuChanged = 1;
        uint32_t iteration = 0;
        while (*gpuChanged > 0)
        {
            *gpuChanged = 0;

            cmdBuf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            cmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
            cmdBuf.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, pipelineLayout, 0, descSet, {});

            PushConsts pc{src_node, num_nodes, bucket, delta, iteration++};

            cmdBuf.pushConstants(
                pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
            cmdBuf.dispatch((num_nodes + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, 1, 1);
            cmdBuf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                   vk::PipelineStageFlagBits::eComputeShader,
                                   vk::DependencyFlags{},
                                   {},
                                   {},
                                   {});
            cmdBuf.end();
            queue.submit(vk::SubmitInfo{0, nullptr, nullptr, 1, &cmdBuf});
            queue.waitIdle();
        }
        
        if (gpuDist[dst_node] != UINT32_MAX) {
            // If the distance is smaller then the current bucket, we have already settled the destination
            if (gpuDist[dst_node] < bucket * delta) {
                break;
            }
        }
    }

    return gpuDist[dst_node];
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

    // Initialize Vulkan
    vk::ApplicationInfo appInfo("DeltaStep", 1, "NoEngine", 1, VK_API_VERSION_1_2);
    vk::Instance instance = vk::createInstance({{}, &appInfo});
    auto physDevices = instance.enumeratePhysicalDevices();
    vk::PhysicalDevice phys = physDevices[0];

    float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo queueInfo({}, 0, 1, &queuePriority);
    vk::Device device = phys.createDevice({{}, 1, &queueInfo});
    auto mem_props = phys.getMemoryProperties();
    vk::Queue queue = device.getQueue(0, 0);

    // Command pool
    vk::CommandPool cmdPool =
        device.createCommandPool({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, 0});


    // Create buffers
    vk::Buffer bufRow =
        gpu::create_exclusive_buffer<uint32_t>(device, graph.first_edges.size(), vk::BufferUsageFlagBits::eStorageBuffer);
    vk::Buffer bufCol =
        gpu::create_exclusive_buffer<uint32_t>(device, graph.targets.size(), vk::BufferUsageFlagBits::eStorageBuffer);
    vk::Buffer bufWeight =
        gpu::create_exclusive_buffer<uint32_t>(device, graph.weights.size(), vk::BufferUsageFlagBits::eStorageBuffer);
    vk::Buffer bufDist =
        gpu::create_exclusive_buffer<uint32_t>(device, graph.num_nodes(), vk::BufferUsageFlagBits::eStorageBuffer);
    vk::Buffer bufChanged = 
        gpu::create_exclusive_buffer<uint32_t>(device, 1, vk::BufferUsageFlagBits::eStorageBuffer);

    vk::DeviceMemory memRow = gpu::alloc_and_bind(device, mem_props, bufRow, vk::MemoryPropertyFlagBits::eHostVisible);
    vk::DeviceMemory memCol = gpu::alloc_and_bind(device, mem_props, bufCol, vk::MemoryPropertyFlagBits::eHostVisible);
    vk::DeviceMemory memWeight = gpu::alloc_and_bind(device, mem_props, bufWeight, vk::MemoryPropertyFlagBits::eHostVisible);
    vk::DeviceMemory memDist = gpu::alloc_and_bind(device, mem_props, bufDist, vk::MemoryPropertyFlagBits::eHostVisible);
    vk::DeviceMemory memChanged = gpu::alloc_and_bind(device, mem_props, bufChanged, vk::MemoryPropertyFlagBits::eHostVisible);

    // Copy graph data
    memcpy(device.mapMemory(memRow, 0, graph.first_edges.size() * sizeof(uint32_t)),
           graph.first_edges.data(),
           graph.first_edges.size() * sizeof(uint32_t));
    memcpy(device.mapMemory(memCol, 0, graph.targets.size() * sizeof(uint32_t)),
           graph.targets.data(),
           graph.targets.size() * sizeof(uint32_t));
    memcpy(device.mapMemory(memWeight, 0, graph.weights.size() * sizeof(uint32_t)),
           graph.weights.data(),
           graph.weights.size() * sizeof(uint32_t));
    device.unmapMemory(memRow);
    device.unmapMemory(memCol);
    device.unmapMemory(memWeight);

    auto gpuDist = (uint32_t *)device.mapMemory(memDist, 0, graph.num_nodes() * sizeof(uint32_t));
    auto gpuChanged = (uint32_t *)device.mapMemory(memChanged, 0, sizeof(uint32_t));

    // Descriptor set layout
    std::vector<vk::DescriptorSetLayoutBinding> bindings(5);
    for (int i = 0; i < 5; i++)
    {
        bindings[i].binding = i;
        bindings[i].descriptorType = vk::DescriptorType::eStorageBuffer;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = vk::ShaderStageFlagBits::eCompute;
    }
    vk::DescriptorSetLayout dsl =
        device.createDescriptorSetLayout({{}, (uint32_t)bindings.size(), bindings.data()});

    // Descriptor pool and set
    vk::DescriptorPoolSize poolSize{vk::DescriptorType::eStorageBuffer, 5};
    vk::DescriptorPool descPool = device.createDescriptorPool({{}, 1, 1, &poolSize});
    vk::DescriptorSet descSet = device.allocateDescriptorSets({descPool, 1, &dsl})[0];

    vk::DescriptorBufferInfo dbiRow{bufRow, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo dbiCol{bufCol, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo dbiWeight{bufWeight, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo dbiDist{bufDist, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo dbiChanged{bufChanged, 0, VK_WHOLE_SIZE};

    std::vector<vk::WriteDescriptorSet> writes = {
        {descSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &dbiRow, nullptr},
        {descSet, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &dbiCol, nullptr},
        {descSet, 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &dbiWeight, nullptr},
        {descSet, 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &dbiDist, nullptr},
        {descSet, 4, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &dbiChanged, nullptr}};
    device.updateDescriptorSets(writes, {});

    // Load SPIR-V shader
    std::vector<uint32_t> spv = common::read_spv("delta_step.spv");
    vk::ShaderModule shader = device.createShaderModule({{}, spv.size() * 4, spv.data()});

    vk::PushConstantRange pcRange{
        vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t) * 2 + sizeof(uint32_t)};
    vk::PipelineLayout pipelineLayout = device.createPipelineLayout({{}, 1, &dsl, 1, &pcRange});

    vk::PipelineShaderStageCreateInfo shaderStage{
        {}, vk::ShaderStageFlagBits::eCompute, shader, "main"};

    vk::Pipeline pipeline =
        device.createComputePipeline({}, {{}, shaderStage, pipelineLayout}).value;

    // Command buffer
    vk::CommandBuffer cmdBuf =
        device.allocateCommandBuffers({cmdPool, vk::CommandBufferLevel::ePrimary, 1})[0];

    common::TimedLogger time_query("Running query with delta " + std::to_string(delta));
    std::uint32_t checksum = 0;
    for (int i = 0; i < 10; i++) {
        checksum ^= run_gpu_sssp(src_nodes[i], dst_nodes[i], graph.num_nodes(), delta, gpuDist, gpuChanged, cmdBuf, queue, pipeline, pipelineLayout, descSet);
    }
    time_query.finished();
    std::cout << "Checksum: " << checksum << std::endl;

    device.unmapMemory(memDist);
    device.unmapMemory(memChanged);

    device.destroyShaderModule(shader);
    device.destroyPipeline(pipeline);
    device.destroyPipelineLayout(pipelineLayout);
    device.destroyDescriptorSetLayout(dsl);
    device.destroyDescriptorPool(descPool);
    device.destroyBuffer(bufRow);
    device.destroyBuffer(bufCol);
    device.destroyBuffer(bufWeight);
    device.destroyBuffer(bufDist);
    device.destroyBuffer(bufChanged);
    device.freeMemory(memRow);
    device.freeMemory(memCol);
    device.freeMemory(memWeight);
    device.freeMemory(memDist);
    device.freeMemory(memChanged);
    device.destroyCommandPool(cmdPool);
    device.destroy();
    instance.destroy();

    return 0;
}
