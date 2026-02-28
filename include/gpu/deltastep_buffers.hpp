#ifndef GPUSSSP_GPU_DELTASTEP_BUFFERS_HPP
#define GPUSSSP_GPU_DELTASTEP_BUFFERS_HPP

#include <array>
#include <cstddef>
#include <cstdint>
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
        // One extra element for max_distance
        buf_dist = gpu::create_exclusive_buffer<uint32_t>(
            device,
            num_nodes + 1,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                vk::BufferUsageFlagBits::eTransferSrc);
        // Host-visible results buffer: [0] = best_distance, [1] = max_distance, [2] =
        // min_changed_id, [3] = max_changed_id
        buf_results = gpu::create_exclusive_buffer<uint32_t>(
            device,
            4,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);
        // Boolean flag arrays
        // These two buffers will be swapped between iterations
        // Two extra elements for min/max changed node IDs
        auto num_blocks = (num_nodes + 31) / 32;
        buf_changed_0 = gpu::create_exclusive_buffer<uint32_t>(
            device,
            num_blocks + 2,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                vk::BufferUsageFlagBits::eTransferSrc);
        buf_changed_1 = gpu::create_exclusive_buffer<uint32_t>(
            device,
            num_blocks + 2,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                vk::BufferUsageFlagBits::eTransferSrc);
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
        mem_dispatch_deltastep = gpu::alloc_and_bind(
            device, mem_props, buf_dispatch_deltastep, vk::MemoryPropertyFlagBits::eDeviceLocal);

        gpu_results =
            static_cast<uint32_t *>(device.mapMemory(mem_results, 0, 4 * sizeof(uint32_t)));
    }

    ~DeltaStepBuffers()
    {
        device.unmapMemory(mem_results);
        device.destroyBuffer(buf_dist);
        device.destroyBuffer(buf_results);
        device.destroyBuffer(buf_changed_0);
        device.destroyBuffer(buf_changed_1);
        device.destroyBuffer(buf_dispatch_deltastep);
        device.freeMemory(mem_dist);
        device.freeMemory(mem_results);
        device.freeMemory(mem_changed_0);
        device.freeMemory(mem_changed_1);
        device.freeMemory(mem_dispatch_deltastep);
    }

    uint32_t *best_distance() { return gpu_results; }
    uint32_t *max_distance() { return gpu_results + 1; }
    uint32_t *min_changed_id() { return gpu_results + 2; }
    uint32_t *max_changed_id() { return gpu_results + 3; }

    [[nodiscard]] std::array<const vk::Buffer, 5> buffers() const
    {
        return {buf_dist, buf_results, buf_changed_0, buf_changed_1, buf_dispatch_deltastep};
    }

  private:
    vk::Buffer buf_dist;
    vk::Buffer buf_results;
    vk::Buffer buf_changed_0;
    vk::Buffer buf_changed_1;
    vk::Buffer buf_dispatch_deltastep;

    vk::DeviceMemory mem_dist;
    vk::DeviceMemory mem_results;
    vk::DeviceMemory mem_changed_0;
    vk::DeviceMemory mem_changed_1;
    vk::DeviceMemory mem_dispatch_deltastep;

    uint32_t *gpu_results;

    vk::Device &device;
};

} // namespace gpusssp::gpu

#endif
