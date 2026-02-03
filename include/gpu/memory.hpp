#ifndef GPUSSSP_MEMORY_HPP
#define GPUSSSP_MEMORY_HPP

#include <vector>
#include <vulkan/vulkan.hpp>

namespace gpusssp::gpu
{
inline auto find_memory_type_index(const vk::PhysicalDeviceMemoryProperties &mem_props,
                                   const vk::MemoryRequirements &mr,
                                   vk::MemoryPropertyFlags flags)
{
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++)
    {
        if ((mr.memoryTypeBits & (1 << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & flags) == flags)
        {
            return i;
        }
    }

    throw std::runtime_error("Couldn't find matching memory");
}

inline auto alloc_and_bind(vk::Device &device,
                           const vk::PhysicalDeviceMemoryProperties &mem_props,
                           const vk::Buffer &buf,
                           vk::MemoryPropertyFlags flags)
{
    vk::MemoryRequirements mr = device.getBufferMemoryRequirements(buf);
    auto mem_type_index = find_memory_type_index(mem_props, mr, flags);
    vk::DeviceMemory mem = device.allocateMemory({mr.size, mem_type_index});
    device.bindBufferMemory(buf, mem, 0);
    return mem;
}

template <typename T>
auto create_exclusive_buffer(vk::Device &device, size_t size, vk::BufferUsageFlags usage)
{
    return device.createBuffer({{}, sizeof(T) * size, usage, vk::SharingMode::eExclusive});
}

struct BufferCopyInfo
{
    const void *src_data;
    vk::Buffer dst_buffer;
    vk::DeviceSize size;
};

inline void copy_buffers_batched(vk::Device &device,
                                 const vk::PhysicalDeviceMemoryProperties &mem_props,
                                 vk::CommandPool command_pool,
                                 vk::Queue queue,
                                 const std::vector<BufferCopyInfo> &copies)
{
    if (copies.empty())
    {
        return;
    }

    std::vector<vk::Buffer> staging_buffers;
    std::vector<vk::DeviceMemory> staging_memories;
    staging_buffers.reserve(copies.size());
    staging_memories.reserve(copies.size());

    for (const auto &copy_info : copies)
    {
        vk::Buffer staging_buffer = device.createBuffer({{},
                                                         copy_info.size,
                                                         vk::BufferUsageFlagBits::eTransferSrc,
                                                         vk::SharingMode::eExclusive});
        staging_buffers.push_back(staging_buffer);

        vk::DeviceMemory staging_memory = alloc_and_bind(
            device,
            mem_props,
            staging_buffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        staging_memories.push_back(staging_memory);

        void *mapped = device.mapMemory(staging_memory, 0, copy_info.size);
        memcpy(mapped, copy_info.src_data, copy_info.size);
        device.unmapMemory(staging_memory);
    }

    vk::CommandBuffer cmd_buffer =
        device.allocateCommandBuffers({command_pool, vk::CommandBufferLevel::ePrimary, 1})[0];

    cmd_buffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    for (size_t i = 0; i < copies.size(); ++i)
    {
        vk::BufferCopy copy_region{0, 0, copies[i].size};
        cmd_buffer.copyBuffer(staging_buffers[i], copies[i].dst_buffer, 1, &copy_region);
    }
    cmd_buffer.end();

    vk::Fence fence = device.createFence({});
    queue.submit({{0, nullptr, nullptr, 1, &cmd_buffer}}, fence);
    (void)device.waitForFences(1, &fence, VK_TRUE, UINT64_MAX);

    device.destroyFence(fence);
    device.freeCommandBuffers(command_pool, 1, &cmd_buffer);
    for (auto buffer : staging_buffers)
    {
        device.destroyBuffer(buffer);
    }
    for (auto memory : staging_memories)
    {
        device.freeMemory(memory);
    }
}

struct DescriptorSetBundle
{
    std::vector<vk::DescriptorSet> descriptor_sets;
    vk::DescriptorSetLayout layout;
    vk::DescriptorPool pool;
};

inline DescriptorSetBundle
create_descriptor_sets(vk::Device &device, const std::vector<std::vector<vk::Buffer>> &buffer_sets)
{
    if (buffer_sets.empty())
    {
        throw std::runtime_error("buffer_sets cannot be empty");
    }

    const uint32_t num_descriptor_sets = static_cast<uint32_t>(buffer_sets.size());
    const uint32_t num_bindings = static_cast<uint32_t>(buffer_sets[0].size());

    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    bindings.reserve(num_bindings);
    for (uint32_t i = 0; i < num_bindings; ++i)
    {
        bindings.push_back(
            {i, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
    }

    vk::DescriptorSetLayout layout =
        device.createDescriptorSetLayout({{}, num_bindings, bindings.data()});

    vk::DescriptorPoolSize pool_size{vk::DescriptorType::eStorageBuffer,
                                     num_bindings * num_descriptor_sets};
    vk::DescriptorPool pool = device.createDescriptorPool({{}, num_descriptor_sets, 1, &pool_size});

    std::vector<vk::DescriptorSetLayout> layouts(num_descriptor_sets, layout);
    std::vector<vk::DescriptorSet> descriptor_sets =
        device.allocateDescriptorSets({pool, num_descriptor_sets, layouts.data()});

    for (uint32_t set_idx = 0; set_idx < num_descriptor_sets; ++set_idx)
    {
        const auto &buffers = buffer_sets[set_idx];
        if (buffers.size() != num_bindings)
        {
            throw std::runtime_error("All descriptor sets must have the same number of buffers");
        }

        std::vector<vk::DescriptorBufferInfo> buffer_infos;
        buffer_infos.reserve(num_bindings);
        for (const auto &buffer : buffers)
        {
            buffer_infos.push_back({buffer, 0, VK_WHOLE_SIZE});
        }

        std::vector<vk::WriteDescriptorSet> writes;
        writes.reserve(num_bindings);
        for (uint32_t binding_idx = 0; binding_idx < num_bindings; ++binding_idx)
        {
            writes.push_back({descriptor_sets[set_idx],
                              binding_idx,
                              0,
                              1,
                              vk::DescriptorType::eStorageBuffer,
                              nullptr,
                              &buffer_infos[binding_idx],
                              nullptr});
        }

        device.updateDescriptorSets(writes, {});
    }

    return {descriptor_sets, layout, pool};
}

} // namespace gpusssp::gpu

#endif
