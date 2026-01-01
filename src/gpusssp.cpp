#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
#include <vulkan/vulkan.hpp>

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

int main(int argc, char **argv)
{
    vk::ApplicationInfo appInfo("DeltaStep", 1, "NoEngine", 1, VK_API_VERSION_1_2);
    vk::Instance instance = vk::createInstance({{}, &appInfo});
    auto physDevices = instance.enumeratePhysicalDevices();
    vk::PhysicalDevice phys = physDevices[0];

    float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo queueInfo({}, 0, 1, &queuePriority);
    vk::Device device = phys.createDevice({{}, 1, &queueInfo});
    vk::Queue queue = device.getQueue(0, 0);

    // 2. Command pool
    vk::CommandPool cmdPool =
        device.createCommandPool({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, 0});

    // 3. Simple example graph (CSR)
    // Graph: 4 vertices, edges with weights
    // 0->1(1), 0->2(4), 1->2(2), 1->3(6), 2->3(3)
    std::vector<uint32_t> row_ptr = {0, 2, 4, 5, 5};
    std::vector<uint32_t> col_idx = {1, 2, 2, 3, 3};
    std::vector<uint32_t> weight = {1, 4, 2, 6, 3};
    uint32_t n = 4;
    uint32_t e = 5;

    std::vector<uint32_t> dist(n, UINT32_MAX);
    dist[0] = 0;

    // 4. Create buffers
    auto createBuffer = [&](vk::DeviceSize size, vk::BufferUsageFlags usage)
    { return device.createBuffer({{}, size, usage, vk::SharingMode::eExclusive}); };
    vk::Buffer bufRow =
        createBuffer(row_ptr.size() * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer);
    vk::Buffer bufCol =
        createBuffer(col_idx.size() * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer);
    vk::Buffer bufWeight =
        createBuffer(weight.size() * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer);
    vk::Buffer bufDist =
        createBuffer(dist.size() * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer);
    vk::Buffer bufChanged = createBuffer(sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer);

    // 5. Allocate memory (host visible for simplicity)
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

    // Copy data
    memcpy(device.mapMemory(memRow, 0, row_ptr.size() * sizeof(uint32_t)),
           row_ptr.data(),
           row_ptr.size() * sizeof(uint32_t));
    memcpy(device.mapMemory(memCol, 0, col_idx.size() * sizeof(uint32_t)),
           col_idx.data(),
           col_idx.size() * sizeof(uint32_t));
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

    // 6. Descriptor set layout
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

    // 7. Descriptor pool and set
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

    // 8. Load SPIR-V shader
    std::vector<uint32_t> spv = readSPV("delta_step.spv");
    vk::ShaderModule shader = device.createShaderModule({{}, spv.size() * 4, spv.data()});

    vk::PushConstantRange pcRange{
        vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t) * 2 + sizeof(uint32_t)};
    vk::PipelineLayout pipelineLayout = device.createPipelineLayout({{}, 1, &dsl, 1, &pcRange});

    vk::PipelineShaderStageCreateInfo shaderStage{
        {}, vk::ShaderStageFlagBits::eCompute, shader, "main"};

    vk::Pipeline pipeline =
        device.createComputePipeline({}, {{}, shaderStage, pipelineLayout}).value;

    // 9. Command buffer
    vk::CommandBuffer cmdBuf =
        device.allocateCommandBuffers({cmdPool, vk::CommandBufferLevel::ePrimary, 1})[0];

    struct PushConsts
    {
        uint32_t n;
        uint32_t bucket_idx;
        uint32_t delta;
    };

    for (uint32_t bucket = 0; bucket < 10; bucket++)
    {
        *gpuChanged = 1;
        uint32_t iter = 0;
        while (*gpuChanged > 0)
        {
            std::cout << bucket << " - " << iter++ << std::endl;
            *gpuChanged = 0;

            cmdBuf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            cmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
            cmdBuf.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, pipelineLayout, 0, descSet, {});

            PushConsts pc{n, bucket, 2}; // delta=2

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

            std::cout << "Distances: ";
            for (uint32_t i = 0; i < n; i++)
                std::cout << gpuDist[i] << " ";
            std::cout << std::endl;
        }
    }

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
