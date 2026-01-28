#ifndef GPUSSSP_GPU_DELTASTEP_BUFFERS_HPP
#define GPUSSSP_GPU_DELTASTEP_BUFFERS_HPP

#include <vulkan/vulkan.hpp>

#include "gpu/memory.hpp"

namespace gpusssp::gpu
{

class DeltaStepBuffers
{
  public:
    DeltaStepBuffers(size_t num_nodes,
                     vk::Device &device,
                     const vk::PhysicalDeviceMemoryProperties &mem_props)
        : num_nodes(num_nodes), device(device)
    {
        // last entry is the maximum overall distance
        buf_dist = gpu::create_exclusive_buffer<uint32_t>(
            device, num_nodes + 1, vk::BufferUsageFlagBits::eStorageBuffer);
        // Boolean flag arrays, last entry is the number of changed nodes
        // These two buffers will be swapped between iterations
        auto num_blocks = (num_nodes + 15) / 16 + 1;
        buf_changed_0 = gpu::create_exclusive_buffer<uint32_t>(
            device, num_blocks, vk::BufferUsageFlagBits::eStorageBuffer);
        buf_changed_1 = gpu::create_exclusive_buffer<uint32_t>(
            device, num_blocks, vk::BufferUsageFlagBits::eStorageBuffer);

        mem_dist = gpu::alloc_and_bind(
            device, mem_props, buf_dist, vk::MemoryPropertyFlagBits::eHostVisible);
        mem_changed_0 = gpu::alloc_and_bind(
            device, mem_props, buf_changed_0, vk::MemoryPropertyFlagBits::eHostVisible);
        mem_changed_1 = gpu::alloc_and_bind(
            device, mem_props, buf_changed_1, vk::MemoryPropertyFlagBits::eHostVisible);

        gpu_dist = (uint32_t *)device.mapMemory(mem_dist, 0, (num_nodes + 1) * sizeof(uint32_t));
        gpu_num_changed_0 = (uint32_t *)device.mapMemory(
            mem_changed_0, (num_blocks - 1) * sizeof(uint32_t), sizeof(uint32_t));

        gpu_num_changed_1 = (uint32_t *)device.mapMemory(
            mem_changed_1, (num_blocks - 1) * sizeof(uint32_t), sizeof(uint32_t));
    }

    ~DeltaStepBuffers()
    {
        device.unmapMemory(mem_dist);
        device.unmapMemory(mem_changed_0);
        device.destroyBuffer(buf_dist);
        device.destroyBuffer(buf_changed_0);
        device.destroyBuffer(buf_changed_1);
        device.freeMemory(mem_dist);
        device.freeMemory(mem_changed_0);
        device.freeMemory(mem_changed_1);
    }

    uint32_t *dist() { return gpu_dist; }

    std::tuple<uint32_t *, uint32_t *> num_changed()
    {
        return std::make_tuple(gpu_num_changed_0, gpu_num_changed_1);
    }

    std::array<const vk::Buffer, 3> buffers() const
    {
        return {buf_dist, buf_changed_0, buf_changed_1};
    }

  private:
    vk::Buffer buf_dist;
    vk::Buffer buf_changed_0;
    vk::Buffer buf_changed_1;

    vk::DeviceMemory mem_dist;
    vk::DeviceMemory mem_changed_0;
    vk::DeviceMemory mem_changed_1;

    uint32_t *gpu_dist;
    uint32_t *gpu_num_changed_0;
    uint32_t *gpu_num_changed_1;

    size_t num_nodes;
    vk::Device &device;
};

} // namespace gpusssp::gpu

#endif
