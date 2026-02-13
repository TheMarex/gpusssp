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
        : device(device)
    {
        // Device-local distance buffer (no longer needs host visibility)
        buf_dist = gpu::create_exclusive_buffer<uint32_t>(
            device, num_nodes, vk::BufferUsageFlagBits::eStorageBuffer);
        // Host-visible results buffer: [0] = best_distance, [1] = max_distance
        buf_results = gpu::create_exclusive_buffer<uint32_t>(
            device, 2, vk::BufferUsageFlagBits::eStorageBuffer);
        // Boolean flag arrays
        // These two buffers will be swapped between iterations
        auto num_blocks = (num_nodes + 31) / 32;
        buf_changed_0 = gpu::create_exclusive_buffer<uint32_t>(
            device, num_blocks, vk::BufferUsageFlagBits::eStorageBuffer);
        buf_changed_1 = gpu::create_exclusive_buffer<uint32_t>(
            device, num_blocks, vk::BufferUsageFlagBits::eStorageBuffer);
        // Min/max of changed node IDs: [0] = min, [1] = max
        buf_min_max_changed_id_0 = gpu::create_exclusive_buffer<uint32_t>(
            device,
            2,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);
        buf_min_max_changed_id_1 = gpu::create_exclusive_buffer<uint32_t>(
            device,
            2,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);
        buf_dispatch_deltastep = gpu::create_exclusive_buffer<uint32_t>(
            device,
            3,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer);

        mem_dist = gpu::alloc_and_bind(
            device, mem_props, buf_dist, vk::MemoryPropertyFlagBits::eDeviceLocal);
        mem_results = gpu::alloc_and_bind(device,
                                          mem_props,
                                          buf_results,
                                          vk::MemoryPropertyFlagBits::eHostVisible |
                                              vk::MemoryPropertyFlagBits::eHostCoherent);
        mem_changed_0 = gpu::alloc_and_bind(
            device, mem_props, buf_changed_0, vk::MemoryPropertyFlagBits::eDeviceLocal);
        mem_changed_1 = gpu::alloc_and_bind(
            device, mem_props, buf_changed_1, vk::MemoryPropertyFlagBits::eDeviceLocal);
        mem_min_max_changed_id_0 = gpu::alloc_and_bind(
            device,
            mem_props,
            buf_min_max_changed_id_0,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        mem_min_max_changed_id_1 = gpu::alloc_and_bind(
            device,
            mem_props,
            buf_min_max_changed_id_1,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        mem_dispatch_deltastep = gpu::alloc_and_bind(
            device, mem_props, buf_dispatch_deltastep, vk::MemoryPropertyFlagBits::eDeviceLocal);

        gpu_results = (uint32_t *)device.mapMemory(mem_results, 0, 2 * sizeof(uint32_t));
        gpu_min_max_changed_id_0 =
            (uint32_t *)device.mapMemory(mem_min_max_changed_id_0, 0, 2 * sizeof(uint32_t));
        gpu_min_max_changed_id_1 =
            (uint32_t *)device.mapMemory(mem_min_max_changed_id_1, 0, 2 * sizeof(uint32_t));
    }

    ~DeltaStepBuffers()
    {
        device.unmapMemory(mem_results);
        device.unmapMemory(mem_min_max_changed_id_0);
        device.unmapMemory(mem_min_max_changed_id_1);
        device.destroyBuffer(buf_dist);
        device.destroyBuffer(buf_results);
        device.destroyBuffer(buf_changed_0);
        device.destroyBuffer(buf_changed_1);
        device.destroyBuffer(buf_min_max_changed_id_0);
        device.destroyBuffer(buf_min_max_changed_id_1);
        device.destroyBuffer(buf_dispatch_deltastep);
        device.freeMemory(mem_dist);
        device.freeMemory(mem_results);
        device.freeMemory(mem_changed_0);
        device.freeMemory(mem_changed_1);
        device.freeMemory(mem_min_max_changed_id_0);
        device.freeMemory(mem_min_max_changed_id_1);
        device.freeMemory(mem_dispatch_deltastep);
    }

    uint32_t *best_distance() { return gpu_results; }
    uint32_t *max_distance() { return gpu_results + 1; }

    uint32_t *min_max_changed_id_0() { return gpu_min_max_changed_id_0; }
    uint32_t *min_max_changed_id_1() { return gpu_min_max_changed_id_1; }

    std::array<const vk::Buffer, 7> buffers() const
    {
        return {buf_dist,
                buf_results,
                buf_changed_0,
                buf_changed_1,
                buf_min_max_changed_id_0,
                buf_min_max_changed_id_1,
                buf_dispatch_deltastep};
    }

  private:
    vk::Buffer buf_dist;
    vk::Buffer buf_results;
    vk::Buffer buf_changed_0;
    vk::Buffer buf_changed_1;
    vk::Buffer buf_min_max_changed_id_0;
    vk::Buffer buf_min_max_changed_id_1;
    vk::Buffer buf_dispatch_deltastep;

    vk::DeviceMemory mem_dist;
    vk::DeviceMemory mem_results;
    vk::DeviceMemory mem_changed_0;
    vk::DeviceMemory mem_changed_1;
    vk::DeviceMemory mem_min_max_changed_id_0;
    vk::DeviceMemory mem_min_max_changed_id_1;
    vk::DeviceMemory mem_dispatch_deltastep;

    uint32_t *gpu_results;
    uint32_t *gpu_min_max_changed_id_0;
    uint32_t *gpu_min_max_changed_id_1;

    vk::Device &device;
};

} // namespace gpusssp::gpu

#endif
