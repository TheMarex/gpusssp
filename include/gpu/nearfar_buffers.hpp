#ifndef GPUSSSP_GPU_NEARFAR_BUFFERS_HPP
#define GPUSSSP_GPU_NEARFAR_BUFFERS_HPP

#include <array>
#include <cstddef>
#include <cstdint>
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
            device,
            num_nodes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);
        buf_results = gpu::create_exclusive_buffer<uint32_t>(
            device,
            5,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);
        buf_near_0 = gpu::create_exclusive_buffer<uint32_t>(
            device,
            num_nodes + 1,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                vk::BufferUsageFlagBits::eTransferSrc);
        buf_near_1 = gpu::create_exclusive_buffer<uint32_t>(
            device,
            num_nodes + 1,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                vk::BufferUsageFlagBits::eTransferSrc);
        buf_far_0 = gpu::create_exclusive_buffer<uint32_t>(
            device,
            num_nodes + 1,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                vk::BufferUsageFlagBits::eTransferSrc);
        buf_far_1 = gpu::create_exclusive_buffer<uint32_t>(
            device,
            num_nodes + 1,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                vk::BufferUsageFlagBits::eTransferSrc);
        buf_dispatch_relax = gpu::create_exclusive_buffer<uint32_t>(
            device,
            3,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer);
        buf_processed = gpu::create_exclusive_buffer<uint32_t>(
            device,
            (num_nodes + 31) / 32,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);
        buf_phase_params = gpu::create_exclusive_buffer<uint32_t>(
            device,
            2,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);

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
        mem_dispatch_relax = gpu::alloc_and_bind(
            device, mem_props, buf_dispatch_relax, vk::MemoryPropertyFlagBits::eDeviceLocal);
        mem_processed = gpu::alloc_and_bind(
            device, mem_props, buf_processed, vk::MemoryPropertyFlagBits::eDeviceLocal);
        mem_phase_params = gpu::alloc_and_bind(
            device, mem_props, buf_phase_params, vk::MemoryPropertyFlagBits::eDeviceLocal);

        gpu_results =
            static_cast<uint32_t *>(device.mapMemory(mem_results, 0, 5 * sizeof(uint32_t)));
    }

    ~NearFarBuffers()
    {
        device.unmapMemory(mem_results);
        device.destroyBuffer(buf_dist);
        device.destroyBuffer(buf_results);
        device.destroyBuffer(buf_near_0);
        device.destroyBuffer(buf_near_1);
        device.destroyBuffer(buf_far_0);
        device.destroyBuffer(buf_far_1);
        device.destroyBuffer(buf_dispatch_relax);
        device.destroyBuffer(buf_processed);
        device.destroyBuffer(buf_phase_params);
        device.freeMemory(mem_dist);
        device.freeMemory(mem_results);
        device.freeMemory(mem_near_0);
        device.freeMemory(mem_near_1);
        device.freeMemory(mem_far_0);
        device.freeMemory(mem_far_1);
        device.freeMemory(mem_dispatch_relax);
        device.freeMemory(mem_processed);
        device.freeMemory(mem_phase_params);
    }

    uint32_t *best_distance() { return gpu_results; }

    uint32_t *num_near() { return gpu_results + 1; }
    uint32_t *num_far() { return gpu_results + 2; }

    uint32_t *gpu_phase() { return gpu_results + 3; }
    uint32_t *gpu_delta() { return gpu_results + 4; }

    [[nodiscard]] std::array<const vk::Buffer, 9> buffers() const
    {
        return {buf_dist,
                buf_results,
                buf_near_0,
                buf_near_1,
                buf_far_0,
                buf_far_1,
                buf_dispatch_relax,
                buf_processed,
                buf_phase_params};
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
    vk::Buffer buf_processed;
    vk::Buffer buf_phase_params;

    vk::DeviceMemory mem_dist;
    vk::DeviceMemory mem_results;
    vk::DeviceMemory mem_near_0;
    vk::DeviceMemory mem_near_1;
    vk::DeviceMemory mem_far_0;
    vk::DeviceMemory mem_far_1;
    vk::DeviceMemory mem_counters;
    vk::DeviceMemory mem_dispatch_relax;
    vk::DeviceMemory mem_processed;
    vk::DeviceMemory mem_phase_params;

    uint32_t *gpu_results;
    uint32_t *gpu_counters{};

    vk::Device &device;
};

} // namespace gpusssp::gpu

#endif
