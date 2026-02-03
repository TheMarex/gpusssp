#ifndef GPUSSSP_GPU_BELLMANFORD_BUFFERS_HPP
#define GPUSSSP_GPU_BELLMANFORD_BUFFERS_HPP

#include <vulkan/vulkan.hpp>

#include "gpu/memory.hpp"

namespace gpusssp::gpu
{

class BellmanFordBuffers
{
  public:
    BellmanFordBuffers(size_t num_nodes,
                       vk::Device &device,
                       const vk::PhysicalDeviceMemoryProperties &mem_props)
        : num_nodes(num_nodes), device(device)
    {
        // Device-local distance buffer
        buf_dist = gpu::create_exclusive_buffer<uint32_t>(
            device, num_nodes, vk::BufferUsageFlagBits::eStorageBuffer);
        // Host-visible results buffer: [0] = best_distance
        buf_results = gpu::create_exclusive_buffer<uint32_t>(
            device, 1, vk::BufferUsageFlagBits::eStorageBuffer);
        // Flag to track if any distance was updated
        buf_changed = gpu::create_exclusive_buffer<uint32_t>(
            device, 1, vk::BufferUsageFlagBits::eStorageBuffer);

        mem_dist = gpu::alloc_and_bind(
            device, mem_props, buf_dist, vk::MemoryPropertyFlagBits::eDeviceLocal);
        mem_results = gpu::alloc_and_bind(
            device, mem_props, buf_results, vk::MemoryPropertyFlagBits::eHostVisible);
        mem_changed = gpu::alloc_and_bind(
            device, mem_props, buf_changed, vk::MemoryPropertyFlagBits::eHostVisible);

        gpu_results = (uint32_t *)device.mapMemory(mem_results, 0, sizeof(uint32_t));
        gpu_changed = (uint32_t *)device.mapMemory(mem_changed, 0, sizeof(uint32_t));
    }

    ~BellmanFordBuffers()
    {
        device.unmapMemory(mem_results);
        device.unmapMemory(mem_changed);
        device.destroyBuffer(buf_dist);
        device.destroyBuffer(buf_results);
        device.destroyBuffer(buf_changed);
        device.freeMemory(mem_dist);
        device.freeMemory(mem_results);
        device.freeMemory(mem_changed);
    }

    uint32_t *best_distance() { return gpu_results; }
    uint32_t *changed() { return gpu_changed; }

    std::array<const vk::Buffer, 3> buffers() const { return {buf_dist, buf_results, buf_changed}; }

  private:
    vk::Buffer buf_dist;
    vk::Buffer buf_results;
    vk::Buffer buf_changed;

    vk::DeviceMemory mem_dist;
    vk::DeviceMemory mem_results;
    vk::DeviceMemory mem_changed;

    uint32_t *gpu_results;
    uint32_t *gpu_changed;

    size_t num_nodes;
    vk::Device &device;
};

} // namespace gpusssp::gpu

#endif
