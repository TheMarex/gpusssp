#ifndef GPUSSSP_GPU_DELTASTEP_BUFFERS_HPP
#define GPUSSSP_GPU_DELTASTEP_BUFFERS_HPP

#include <vulkan/vulkan.hpp>

#include "gpu/memory.hpp"

namespace gpusssp::gpu
{

class DeltaStepBuffers
{
  public:
    DeltaStepBuffers(size_t num_nodes, vk::Device &device) : num_nodes(num_nodes), device(device) {}

    ~DeltaStepBuffers()
    {
        device.unmapMemory(mem_dist);
        device.unmapMemory(mem_was_changed);
        device.destroyBuffer(buf_dist);
        device.destroyBuffer(buf_was_changed);
        device.destroyBuffer(buf_is_changed);
        device.freeMemory(mem_dist);
        device.freeMemory(mem_is_changed);
        device.freeMemory(mem_was_changed);
    }

    void initialize(const vk::PhysicalDeviceMemoryProperties &mem_props)
    {
        // last entry is the maximum overall distance
        buf_dist = gpu::create_exclusive_buffer<uint32_t>(
            device, num_nodes + 1, vk::BufferUsageFlagBits::eStorageBuffer);
        // Boolean flag arrays, last entry is the number of changed nodes
        auto num_blocks = (num_nodes + 15 / 16) + 1;
        buf_is_changed = gpu::create_exclusive_buffer<uint32_t>(
            device, num_blocks, vk::BufferUsageFlagBits::eStorageBuffer);
        buf_was_changed = gpu::create_exclusive_buffer<uint32_t>(
            device, num_blocks, vk::BufferUsageFlagBits::eStorageBuffer);

        mem_dist = gpu::alloc_and_bind(
            device, mem_props, buf_dist, vk::MemoryPropertyFlagBits::eHostVisible);
        mem_is_changed = gpu::alloc_and_bind(
            device, mem_props, buf_is_changed, vk::MemoryPropertyFlagBits::eHostVisible);
        mem_was_changed = gpu::alloc_and_bind(
            device, mem_props, buf_was_changed, vk::MemoryPropertyFlagBits::eHostVisible);

        gpu_dist = (uint32_t *)device.mapMemory(mem_dist, 0, (num_nodes + 1) * sizeof(uint32_t));
        gpu_num_changed = (uint32_t *)device.mapMemory(
            mem_is_changed, (num_blocks - 1) * sizeof(uint32_t), sizeof(uint32_t));
    }

    uint32_t *dist() { return gpu_dist; }

    uint32_t *num_changed() { return gpu_num_changed; }

    std::array<const vk::Buffer, 3> buffers() const
    {
        return {buf_dist, buf_is_changed, buf_was_changed};
    }

  private:
    vk::Buffer buf_dist;
    vk::Buffer buf_is_changed;
    vk::Buffer buf_was_changed;

    vk::DeviceMemory mem_dist;
    vk::DeviceMemory mem_is_changed;
    vk::DeviceMemory mem_was_changed;

    uint32_t *gpu_dist;
    uint32_t *gpu_num_changed;

    size_t num_nodes;
    vk::Device &device;
};

} // namespace gpusssp::gpu

#endif
