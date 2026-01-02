#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <vulkan/vulkan.hpp>

#include "common/files.hpp"
#include "common/shader.hpp"
#include "common/timed_logger.hpp"
#include "common/coordinate.hpp"
#include "common/weighted_graph.hpp"
#include "common/nearest_neighbour.hpp"

using namespace gpusssp;

const uint32_t WORKGROUP_SIZE = 128;

common::Coordinate string_to_coordinate(const std::string& s) {
    auto pos = s.find(',');
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
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] 
                  << " <graph_path> SRC_LON,SRC_LAT DEST_LON,DEST_LAT [DELTA]" << std::endl;
        std::cerr << "Example: " << argv[0] 
                  << " cache/berlin 13.3889,52.5170 13.4050,52.5200 3600" << std::endl;
        return 1;
    }
    
    std::string graph_path = argv[1];
    auto src_coord = string_to_coordinate(argv[2]);
    auto dst_coord = string_to_coordinate(argv[3]);

    auto delta = 3600u;

    if (argc == 5) {
      delta = std::stoi(argv[4]);
    }
    
    std::cout << "Loading graph from: " << graph_path << std::endl;
    
    // Load graph and coordinates
    auto graph = common::files::read_weighted_graph<uint32_t>(graph_path);
    auto coordinates = common::files::read_coordinates(graph_path);
    
    std::cout << "Graph loaded: " << graph.num_nodes() << " nodes, " 
              << graph.num_edges() << " edges" << std::endl;
    
    // Find closest nodes
    common::NearestNeighbour nn(coordinates);
    uint32_t src_node = nn.nearest(src_coord);
    uint32_t dst_node = nn.nearest(dst_coord);
    
    std::cout << "Source: " << src_coord << " -> Node " << src_node << " " << coordinates[src_node] << std::endl;
    std::cout << "Target: " << dst_coord << " -> Node " << dst_node << " " << coordinates[dst_node] << std::endl;

    // Initialize Vulkan
    vk::ApplicationInfo appInfo("DeltaStep", 1, "NoEngine", 1, VK_API_VERSION_1_2);
    vk::Instance instance = vk::createInstance({{}, &appInfo});
    auto physDevices = instance.enumeratePhysicalDevices();
    vk::PhysicalDevice phys = physDevices[0];

    float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo queueInfo({}, 0, 1, &queuePriority);
    vk::Device device = phys.createDevice({{}, 1, &queueInfo});
    vk::Queue queue = device.getQueue(0, 0);

    // Command pool
    vk::CommandPool cmdPool =
        device.createCommandPool({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, 0});


    // Create buffers
    auto createBuffer = [&](vk::DeviceSize size, vk::BufferUsageFlags usage)
    { return device.createBuffer({{}, size, usage, vk::SharingMode::eExclusive}); };
    vk::Buffer bufRow =
        createBuffer(graph.first_edges.size() * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer);
    vk::Buffer bufCol =
        createBuffer(graph.targets.size() * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer);
    vk::Buffer bufWeight =
        createBuffer(graph.weights.size() * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer);
    vk::Buffer bufDist =
        createBuffer(graph.num_nodes() * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer);
    vk::Buffer bufChanged = createBuffer(sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer);

    // Allocate memory (host visible for simplicity)
    auto allocAndBind = [&](vk::Buffer buf)
    {
        vk::MemoryRequirements mr = device.getBufferMemoryRequirements(buf);
        uint32_t memTypeIndex = 0;
        vk::PhysicalDeviceMemoryProperties memProps = phys.getMemoryProperties();
        for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
        {
            if ((mr.memoryTypeBits & (1 << i)) &&
                (memProps.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible))
            {
                memTypeIndex = i;
                break;
            }
        }
        vk::DeviceMemory mem = device.allocateMemory({mr.size, memTypeIndex});
        device.bindBufferMemory(buf, mem, 0);
        return mem;
    };

    vk::DeviceMemory memRow = allocAndBind(bufRow);
    vk::DeviceMemory memCol = allocAndBind(bufCol);
    vk::DeviceMemory memWeight = allocAndBind(bufWeight);
    vk::DeviceMemory memDist = allocAndBind(bufDist);
    vk::DeviceMemory memChanged = allocAndBind(bufChanged);

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
    for (int i = 0; i < 10; i++) {
        auto result = run_gpu_sssp(src_node, dst_node, graph.num_nodes(), delta, gpuDist, gpuChanged, cmdBuf, queue, pipeline, pipelineLayout, descSet);
        std::cout << src_node << "->" << dst_node << " => " << result << std::endl;
    }
    time_query.finished();

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
