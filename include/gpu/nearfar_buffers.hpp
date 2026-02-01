#ifndef GPUSSSP_GPU_NEARFAR_BUFFERS_HPP
#define GPUSSSP_GPU_NEARFAR_BUFFERS_HPP

#include <vulkan/vulkan.hpp>

#include "gpu/memory.hpp"

namespace gpusssp::gpu
{

class NearFarBuffers
{
  public:
    NearFarBuffers(size_t num_nodes,
                   vk::Device &device,
                   const vk::PhysicalDeviceMemoryProperties &mem_props)
        : device(device)
    {
        buf_dist = gpu::create_exclusive_buffer<uint32_t>(
            device, num_nodes, vk::BufferUsageFlagBits::eStorageBuffer);
        buf_results = gpu::create_exclusive_buffer<uint32_t>(
            device, 1, vk::BufferUsageFlagBits::eStorageBuffer);
        buf_near_0 = gpu::create_exclusive_buffer<uint32_t>(
            device, num_nodes, vk::BufferUsageFlagBits::eStorageBuffer);
        buf_near_1 = gpu::create_exclusive_buffer<uint32_t>(
            device, num_nodes, vk::BufferUsageFlagBits::eStorageBuffer);
        buf_far_0 = gpu::create_exclusive_buffer<uint32_t>(
            device, num_nodes, vk::BufferUsageFlagBits::eStorageBuffer);
        buf_far_1 = gpu::create_exclusive_buffer<uint32_t>(
            device, num_nodes, vk::BufferUsageFlagBits::eStorageBuffer);
        buf_counters = gpu::create_exclusive_buffer<uint32_t>(
            device, 4, vk::BufferUsageFlagBits::eStorageBuffer);
        buf_dispatch_relax = gpu::create_exclusive_buffer<uint32_t>(
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
        mem_near_0 = gpu::alloc_and_bind(
            device, mem_props, buf_near_0, vk::MemoryPropertyFlagBits::eDeviceLocal);
        mem_near_1 = gpu::alloc_and_bind(
            device, mem_props, buf_near_1, vk::MemoryPropertyFlagBits::eDeviceLocal);
        mem_far_0 = gpu::alloc_and_bind(
            device, mem_props, buf_far_0, vk::MemoryPropertyFlagBits::eDeviceLocal);
        mem_far_1 = gpu::alloc_and_bind(
            device, mem_props, buf_far_1, vk::MemoryPropertyFlagBits::eDeviceLocal);
        mem_counters = gpu::alloc_and_bind(device,
                                           mem_props,
                                           buf_counters,
                                           vk::MemoryPropertyFlagBits::eHostVisible |
                                               vk::MemoryPropertyFlagBits::eHostCoherent);
        mem_dispatch_relax = gpu::alloc_and_bind(
            device, mem_props, buf_dispatch_relax, vk::MemoryPropertyFlagBits::eDeviceLocal);

        gpu_results = (uint32_t *)device.mapMemory(mem_results, 0, sizeof(uint32_t));
        gpu_counters = (uint32_t *)device.mapMemory(mem_counters, 0, 4 * sizeof(uint32_t));
    }

    ~NearFarBuffers()
    {
        device.unmapMemory(mem_results);
        device.unmapMemory(mem_counters);
        device.destroyBuffer(buf_dist);
        device.destroyBuffer(buf_results);
        device.destroyBuffer(buf_near_0);
        device.destroyBuffer(buf_near_1);
        device.destroyBuffer(buf_far_0);
        device.destroyBuffer(buf_far_1);
        device.destroyBuffer(buf_counters);
        device.destroyBuffer(buf_dispatch_relax);
        device.freeMemory(mem_dist);
        device.freeMemory(mem_results);
        device.freeMemory(mem_near_0);
        device.freeMemory(mem_near_1);
        device.freeMemory(mem_far_0);
        device.freeMemory(mem_far_1);
        device.freeMemory(mem_counters);
        device.freeMemory(mem_dispatch_relax);
    }

    uint32_t *best_distance() { return gpu_results; }

    uint32_t *num_near() { return gpu_counters; }
    uint32_t *num_next_near() { return gpu_counters + 1; }
    uint32_t *num_far() { return gpu_counters + 2; }
    uint32_t *num_next_far() { return gpu_counters + 3; }

    std::array<const vk::Buffer, 8> buffers() const
    {
        return {buf_dist,
                buf_results,
                buf_near_0,
                buf_near_1,
                buf_far_0,
                buf_far_1,
                buf_counters,
                buf_dispatch_relax};
    }

  private:
    vk::Buffer buf_dist;
    vk::Buffer buf_results;
    vk::Buffer buf_near_0;
    vk::Buffer buf_near_1;
    vk::Buffer buf_far_0;
    vk::Buffer buf_far_1;
    vk::Buffer buf_counters;
    vk::Buffer buf_dispatch_relax;

    vk::DeviceMemory mem_dist;
    vk::DeviceMemory mem_results;
    vk::DeviceMemory mem_near_0;
    vk::DeviceMemory mem_near_1;
    vk::DeviceMemory mem_far_0;
    vk::DeviceMemory mem_far_1;
    vk::DeviceMemory mem_counters;
    vk::DeviceMemory mem_dispatch_relax;

    uint32_t *gpu_results;
    uint32_t *gpu_counters;

    vk::Device &device;
};

} // namespace gpusssp::gpu

#endif
