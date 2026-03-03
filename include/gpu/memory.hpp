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

inline auto alloc_and_bind(vk::Device device,
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
auto create_exclusive_buffer(vk::Device device, size_t size, vk::BufferUsageFlags usage)
{
    return device.createBuffer({{}, sizeof(T) * size, usage, vk::SharingMode::eExclusive});
}

struct BufferCopyInfo
{
    const void *src_data;
    vk::Buffer dst_buffer;
    vk::DeviceSize size;
};

template <typename QueueT = vk::Queue>
inline void copy_buffers_batched(vk::Device device,
                                 const vk::PhysicalDeviceMemoryProperties &mem_props,
                                 vk::CommandPool command_pool,
                                 QueueT queue,
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

} // namespace gpusssp::gpu

#endif
