#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <vulkan/vulkan.hpp>

#include "common/files.hpp"
#include "common/coordinate.hpp"
#include "common/weighted_graph.hpp"

using namespace gpusssp::common;

const uint32_t WORKGROUP_SIZE = 128;

// Utility: read SPIR-V file
std::vector<uint32_t> readSPV(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    assert(file.is_open());
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    size_t wordCount = bytes.size() / 4;
    std::vector<uint32_t> spv(wordCount);
    memcpy(spv.data(), bytes.data(), wordCount * 4);
    return spv;
}

// Find the closest node to a given coordinate
uint32_t find_closest_node(const std::vector<Coordinate> &coordinates, 
                           const Coordinate& target)
{
    uint32_t closest_node = 0;
    long min_dist_sq = std::numeric_limits<long>::max();
    
    for (uint32_t i = 0; i < coordinates.size(); ++i) {
        long dist_sq = euclid_squared_distance(coordinates[i], target);
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            closest_node = i;
        }
    }
    
    return closest_node;
}

Coordinate string_to_coordinate(const std::string& s) {
    auto pos = s.find(',');
    return Coordinate::from_floating(std::stod(s.substr(0, pos)), std::stod(s.substr(pos+1)));
}

int main(int argc, char **argv)
{
    // Parse command line arguments
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] 
                  << " <graph_path> SRC_LON,SRC_LAT DEST_LON,DEST_LAT" << std::endl;
        std::cerr << "Example: " << argv[0] 
                  << " cache/berlin 13.3889,52.5170 13.4050,52.5200" << std::endl;
        return 1;
    }
    
    std::string graph_path = argv[1];
    auto src_coord = string_to_coordinate(argv[2]);
    auto dst_coord = string_to_coordinate(argv[3]);
    
    std::cout << "Loading graph from: " << graph_path << std::endl;
    
    // Load graph and coordinates
    auto graph = files::read_weighted_graph<uint32_t>(graph_path);
    auto coordinates = files::read_coordinates(graph_path);
    
    std::cout << "Graph loaded: " << graph.num_nodes() << " nodes, " 
              << graph.num_edges() << " edges" << std::endl;
    
    // Find closest nodes
    uint32_t src_node = find_closest_node(coordinates, src_coord);
    uint32_t dst_node = find_closest_node(coordinates, dst_coord);
    
    std::cout << "Source: " << src_coord << " -> Node " << src_node << " " << coordinates[src_node] << std::endl;
    std::cout << "Target: " << dst_coord << " -> Node " << dst_node << " " << coordinates[dst_node] << std::endl;
    
    // Extract graph data in CSR format
    auto [first_out, head, weight] = WeightedGraph<uint32_t>::unwrap(graph);
    uint32_t n = graph.num_nodes();
    
    std::cout << "Computing SSSP from node " << src_node << "..." << std::endl;

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

    // Initialize distances
    std::vector<uint32_t> dist(n, UINT32_MAX);
    dist[src_node] = 0;

    // Create buffers
    auto createBuffer = [&](vk::DeviceSize size, vk::BufferUsageFlags usage)
    { return device.createBuffer({{}, size, usage, vk::SharingMode::eExclusive}); };
    vk::Buffer bufRow =
        createBuffer(first_out.size() * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer);
    vk::Buffer bufCol =
        createBuffer(head.size() * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer);
    vk::Buffer bufWeight =
        createBuffer(weight.size() * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer);
    vk::Buffer bufDist =
        createBuffer(dist.size() * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer);
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
    memcpy(device.mapMemory(memRow, 0, first_out.size() * sizeof(uint32_t)),
           first_out.data(),
           first_out.size() * sizeof(uint32_t));
    memcpy(device.mapMemory(memCol, 0, head.size() * sizeof(uint32_t)),
           head.data(),
           head.size() * sizeof(uint32_t));
    memcpy(device.mapMemory(memWeight, 0, weight.size() * sizeof(uint32_t)),
           weight.data(),
           weight.size() * sizeof(uint32_t));
    device.unmapMemory(memRow);
    device.unmapMemory(memCol);
    device.unmapMemory(memWeight);

    auto gpuDist = (uint32_t *)device.mapMemory(memDist, 0, dist.size() * sizeof(uint32_t));
    memcpy(gpuDist, dist.data(), dist.size() * sizeof(uint32_t));
    auto gpuChanged = (uint32_t *)device.mapMemory(memChanged, 0, sizeof(uint32_t));
    *gpuChanged = 0;

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
    std::vector<uint32_t> spv = readSPV("delta_step.spv");
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

    struct PushConsts
    {
        uint32_t n;
        uint32_t bucket_idx;
        uint32_t delta;
    };

    const uint32_t DELTA = 1000; // Adjust based on edge weight scale
    const uint32_t NUM_BUCKETS = 1000; // Process up to DELTA * NUM_BUCKETS
    
    // Run delta-stepping algorithm
    for (uint32_t bucket = 0; bucket < NUM_BUCKETS; bucket++)
    {
        *gpuChanged = 1;
        uint32_t iter = 0;
        while (*gpuChanged > 0)
        {
            if (iter == 0) {
                std::cout << "Processing bucket " << bucket << "..." << std::endl;
            }
            *gpuChanged = 0;

            cmdBuf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            cmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
            cmdBuf.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, pipelineLayout, 0, descSet, {});

            PushConsts pc{n, bucket, DELTA};

            cmdBuf.pushConstants(
                pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
            cmdBuf.dispatch((n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, 1, 1);
            cmdBuf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                   vk::PipelineStageFlagBits::eComputeShader,
                                   vk::DependencyFlags{},
                                   {},
                                   {},
                                   {});
            cmdBuf.end();
            queue.submit(vk::SubmitInfo{0, nullptr, nullptr, 1, &cmdBuf});
            queue.waitIdle();
            
            iter++;
        }
        
        // Early termination: if target is reached and stable
        if (gpuDist[dst_node] != UINT32_MAX) {
            // Check if we've processed all distances up to the target
            if (gpuDist[dst_node] < bucket * DELTA) {
                std::cout << "Target reached at bucket " << bucket << std::endl;
                break;
            }
        }
    }

    // Output results
    std::cout << "\n=== SSSP Results ===" << std::endl;
    std::cout << "Distance from node " << src_node << " to node " << dst_node << ": ";
    if (gpuDist[dst_node] == UINT32_MAX) {
        std::cout << "UNREACHABLE" << std::endl;
    } else {
        std::cout << gpuDist[dst_node] << std::endl;
        
        // Calculate real-world distance
        double haversine_dist = haversine_distance(coordinates[src_node], coordinates[dst_node]);
        std::cout << "Haversine distance: " << haversine_dist << " meters" << std::endl;
    }
    std::cout << std::endl;

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
}
