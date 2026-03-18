#ifndef GPUSSSP_GPU_DEBUG_HPP
#define GPUSSSP_GPU_DEBUG_HPP

#include "gpu/memory.hpp"

#include <vulkan/vulkan.hpp>

namespace gpusssp::gpu
{

template <typename T>
std::vector<T> read_buffer(vk::Device device,
                           const vk::PhysicalDeviceMemoryProperties &mem_props,
                           vk::CommandPool cmd_pool,
                           vk::Queue queue,
                           vk::Buffer buffer,
                           size_t num_elements)
{
    auto staging_buffer = gpu::create_exclusive_buffer<uint32_t>(
        device, num_elements, vk::BufferUsageFlagBits::eTransferDst);

    auto staging_memory = gpu::alloc_and_bind(device,
                                              mem_props,
                                              staging_buffer,
                                              vk::MemoryPropertyFlagBits::eHostVisible |
                                                  vk::MemoryPropertyFlagBits::eHostCoherent);

    auto cmd_bufs = device.allocateCommandBuffers({cmd_pool, vk::CommandBufferLevel::ePrimary, 1});
    auto &cmd_buf = cmd_bufs[0];

    cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    cmd_buf.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eTransfer,
        vk::DependencyFlags{},
        vk::MemoryBarrier{vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead},
        {},
        {});

    vk::BufferCopy copy_region{0, 0, num_elements * sizeof(uint32_t)};
    cmd_buf.copyBuffer(buffer, staging_buffer, 1, &copy_region);

    cmd_buf.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eHost,
        vk::DependencyFlags{},
        vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead},
        {},
        {});

    cmd_buf.end();

    vk::Fence fence = device.createFence({});
    queue.submit({{0, nullptr, nullptr, 1, &cmd_buf}}, fence);
    (void)device.waitForFences(1, &fence, VK_TRUE, UINT64_MAX);

    void *mapped = device.mapMemory(staging_memory, 0, num_elements * sizeof(uint32_t));
    std::vector<T> data(num_elements);
    std::memcpy(data.data(), mapped, num_elements * sizeof(uint32_t));
    device.unmapMemory(staging_memory);

    device.destroyFence(fence);
    device.freeCommandBuffers(cmd_pool, 1, &cmd_buf);
    device.destroyBuffer(staging_buffer);
    device.freeMemory(staging_memory);

    return data;
}
} // namespace gpusssp::gpu

#endif
