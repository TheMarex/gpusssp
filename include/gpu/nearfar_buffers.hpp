#ifndef GPUSSSP_GPU_NEARFAR_BUFFERS_HPP
#define GPUSSSP_GPU_NEARFAR_BUFFERS_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <vulkan/vulkan.hpp>

#include "common/constants.hpp"
#include "gpu/memory.hpp"

namespace gpusssp::gpu
{

class NearFarBuffers
{
  public:
    NearFarBuffers(size_t num_nodes,
                   vk::Device device,
                   const vk::PhysicalDeviceMemoryProperties &mem_props)
        : num_nodes(static_cast<uint32_t>(num_nodes)), device(device)
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

    [[nodiscard]] vk::Buffer dist_buffer() const { return buf_dist; }
    [[nodiscard]] std::array<vk::Buffer, 2> near_buffers() const
    {
        return {buf_near_0, buf_near_1};
    }
    [[nodiscard]] std::array<vk::Buffer, 2> far_buffers() const { return {buf_far_0, buf_far_1}; }
    [[nodiscard]] vk::Buffer dispatch_buffer() const { return buf_dispatch_relax; }
    [[nodiscard]] vk::Buffer processed_buffer() const { return buf_processed; }
    [[nodiscard]] vk::Buffer phase_params_buffer() const { return buf_phase_params; }

    void cmd_init_dist(vk::CommandBuffer cmd_buf, uint32_t src_node)
    {
        cmd_buf.fillBuffer(buf_dist, 0, num_nodes * sizeof(uint32_t), common::INF_WEIGHT);
        cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                vk::PipelineStageFlagBits::eTransfer,
                                vk::DependencyFlags{},
                                vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite,
                                                  vk::AccessFlagBits::eTransferWrite},
                                {},
                                {});
        uint32_t zero = 0;
        cmd_buf.updateBuffer(buf_dist, src_node * sizeof(uint32_t), sizeof(uint32_t), &zero);
    }

    void cmd_init_near_far(vk::CommandBuffer cmd_buf, uint32_t src_node)
    {
        uint32_t zero = 0;
        uint32_t one = 1;
        cmd_buf.updateBuffer(buf_near_0, 0, sizeof(uint32_t), &src_node);
        cmd_buf.updateBuffer(buf_near_0, num_nodes * sizeof(uint32_t), sizeof(uint32_t), &one);
        cmd_buf.updateBuffer(buf_far_0, num_nodes * sizeof(uint32_t), sizeof(uint32_t), &zero);
    }

    void cmd_init_phase_params(vk::CommandBuffer cmd_buf, uint32_t delta)
    {
        uint32_t zero = 0;
        cmd_buf.updateBuffer(buf_phase_params, 0, sizeof(uint32_t), &zero);
        cmd_buf.updateBuffer(buf_phase_params, sizeof(uint32_t), sizeof(uint32_t), &delta);
    }

    void cmd_clear_near(vk::CommandBuffer cmd_buf, uint32_t buffer_idx)
    {
        auto &buf = buffer_idx == 0 ? buf_near_0 : buf_near_1;
        cmd_buf.fillBuffer(buf, num_nodes * sizeof(uint32_t), sizeof(uint32_t), 0);
    }

    void cmd_clear_far_and_processed(vk::CommandBuffer cmd_buf, uint32_t current_far_buffer_idx)
    {
        auto &next_far = current_far_buffer_idx == 0 ? buf_far_1 : buf_far_0;
        cmd_buf.fillBuffer(buf_near_0, num_nodes * sizeof(uint32_t), sizeof(uint32_t), 0);
        cmd_buf.fillBuffer(next_far, num_nodes * sizeof(uint32_t), sizeof(uint32_t), 0);
        cmd_buf.fillBuffer(buf_processed, 0, VK_WHOLE_SIZE, 0);
    }

    void cmd_sync_dist(vk::CommandBuffer cmd_buf, uint32_t dst_node)
    {
        vk::BufferCopy copy{dst_node * sizeof(uint32_t), 0, sizeof(uint32_t)};
        cmd_buf.copyBuffer(buf_dist, buf_results, 1, &copy);
    }

    void cmd_sync_near_count(vk::CommandBuffer cmd_buf, uint32_t buffer_idx)
    {
        auto &buf = buffer_idx == 0 ? buf_near_0 : buf_near_1;
        vk::BufferCopy copy{num_nodes * sizeof(uint32_t), 1 * sizeof(uint32_t), sizeof(uint32_t)};
        cmd_buf.copyBuffer(buf, buf_results, 1, &copy);
    }

    void cmd_sync_far_count(vk::CommandBuffer cmd_buf, uint32_t buffer_idx)
    {
        auto &buf = buffer_idx == 0 ? buf_far_0 : buf_far_1;
        vk::BufferCopy copy{num_nodes * sizeof(uint32_t), 2 * sizeof(uint32_t), sizeof(uint32_t)};
        cmd_buf.copyBuffer(buf, buf_results, 1, &copy);
    }

    void cmd_sync_phase_params(vk::CommandBuffer cmd_buf)
    {
        vk::BufferCopy phase_copy{0, 3 * sizeof(uint32_t), sizeof(uint32_t)};
        vk::BufferCopy delta_copy{sizeof(uint32_t), 4 * sizeof(uint32_t), sizeof(uint32_t)};
        cmd_buf.copyBuffer(buf_phase_params, buf_results, 1, &phase_copy);
        cmd_buf.copyBuffer(buf_phase_params, buf_results, 1, &delta_copy);
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

    uint32_t *gpu_results = nullptr;
    uint32_t *gpu_counters = nullptr;

    uint32_t num_nodes = 0;
    vk::Device device;
};

} // namespace gpusssp::gpu

#endif
