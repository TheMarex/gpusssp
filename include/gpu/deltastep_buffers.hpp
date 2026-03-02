#ifndef GPUSSSP_GPU_DELTASTEP_BUFFERS_HPP
#define GPUSSSP_GPU_DELTASTEP_BUFFERS_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <vulkan/vulkan.hpp>

#include "common/constants.hpp"
#include "gpu/memory.hpp"

namespace gpusssp::gpu
{

class DeltaStepBuffers
{
  public:
    DeltaStepBuffers(size_t num_nodes,
                     vk::Device &device,
                     const vk::PhysicalDeviceMemoryProperties &mem_props)
        : num_nodes(static_cast<uint32_t>(num_nodes)),
          num_blocks(static_cast<uint32_t>((num_nodes + 31) / 32)), device(device)
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
        buf_params = gpu::create_exclusive_buffer<uint32_t>(
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
        mem_changed_0 = gpu::alloc_and_bind(
            device, mem_props, buf_changed_0, vk::MemoryPropertyFlagBits::eDeviceLocal);
        mem_changed_1 = gpu::alloc_and_bind(
            device, mem_props, buf_changed_1, vk::MemoryPropertyFlagBits::eDeviceLocal);
        mem_dispatch_deltastep = gpu::alloc_and_bind(
            device, mem_props, buf_dispatch_deltastep, vk::MemoryPropertyFlagBits::eDeviceLocal);
        mem_params = gpu::alloc_and_bind(
            device, mem_props, buf_params, vk::MemoryPropertyFlagBits::eDeviceLocal);

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
        device.destroyBuffer(buf_params);
        device.freeMemory(mem_dist);
        device.freeMemory(mem_results);
        device.freeMemory(mem_changed_0);
        device.freeMemory(mem_changed_1);
        device.freeMemory(mem_dispatch_deltastep);
        device.freeMemory(mem_params);
    }

    uint32_t *best_distance() { return gpu_results; }
    uint32_t *max_distance() { return gpu_results + 1; }
    uint32_t *min_changed_id() { return gpu_results + 2; }
    uint32_t *max_changed_id() { return gpu_results + 3; }

    [[nodiscard]] vk::Buffer dist_buffer() const { return buf_dist; }
    [[nodiscard]] std::array<vk::Buffer, 2> changed_buffers() const
    {
        return {buf_changed_0, buf_changed_1};
    }
    [[nodiscard]] vk::Buffer dispatch_buffer() const { return buf_dispatch_deltastep; }
    [[nodiscard]] vk::Buffer results_buffer() const { return buf_results; }
    [[nodiscard]] vk::Buffer params_buffer() const { return buf_params; }

    void cmd_init_dist(vk::CommandBuffer &cmd_buf, uint32_t src_node)
    {
        cmd_buf.fillBuffer(buf_dist, 0, (num_nodes + 1) * sizeof(uint32_t), common::INF_WEIGHT);
        cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                vk::PipelineStageFlagBits::eTransfer,
                                vk::DependencyFlags{},
                                vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite,
                                                  vk::AccessFlagBits::eTransferWrite},
                                {},
                                {});
        uint32_t zero = 0;
        cmd_buf.updateBuffer(buf_dist, src_node * sizeof(uint32_t), sizeof(uint32_t), &zero);
        cmd_buf.updateBuffer(buf_dist, num_nodes * sizeof(uint32_t), sizeof(uint32_t), &zero);
    }

    void cmd_init_changed(vk::CommandBuffer &cmd_buf, uint32_t buffer_idx)
    {
        auto &buf = buffer_idx == 0 ? buf_changed_0 : buf_changed_1;
        uint32_t first_pass_min_max[] = {0, num_nodes - 1};
        cmd_buf.fillBuffer(buf, 0, num_blocks * sizeof(uint32_t), UINT32_MAX);
        cmd_buf.updateBuffer(buf, num_blocks * sizeof(uint32_t), 8, first_pass_min_max);
    }

    void cmd_clear_changed(vk::CommandBuffer &cmd_buf, uint32_t buffer_idx)
    {
        auto &buf = buffer_idx == 0 ? buf_changed_0 : buf_changed_1;
        const uint32_t min_max_init[] = {num_nodes, 0};
        cmd_buf.fillBuffer(buf, 0, num_blocks * sizeof(uint32_t), 0);
        cmd_buf.updateBuffer(buf, num_blocks * sizeof(uint32_t), 8, min_max_init);
    }

    void cmd_update_params(vk::CommandBuffer &cmd_buf, uint32_t bucket_idx, uint32_t delta)
    {
        cmd_buf.updateBuffer(buf_params, 0, sizeof(uint32_t), &bucket_idx);
        cmd_buf.updateBuffer(buf_params, sizeof(uint32_t), sizeof(uint32_t), &delta);
    }

    void cmd_sync_results(vk::CommandBuffer &cmd_buf, uint32_t dst_node, uint32_t source_changed_buffer_idx)
    {
        auto &changed_buf =
            source_changed_buffer_idx == 0 ? buf_changed_0 : buf_changed_1;
        const std::array<vk::BufferCopy, 2> dist_copy = {
            vk::BufferCopy{dst_node * sizeof(uint32_t), 0, sizeof(uint32_t)},
            vk::BufferCopy{num_nodes * sizeof(uint32_t), sizeof(uint32_t), sizeof(uint32_t)}};
        vk::BufferCopy min_max_copy{num_blocks * sizeof(uint32_t), 2 * sizeof(uint32_t), 8};
        cmd_buf.copyBuffer(buf_dist, buf_results, dist_copy);
        cmd_buf.copyBuffer(changed_buf, buf_results, 1, &min_max_copy);
    }

  private:
    vk::Buffer buf_dist;
    vk::Buffer buf_results;
    vk::Buffer buf_changed_0;
    vk::Buffer buf_changed_1;
    vk::Buffer buf_dispatch_deltastep;
    vk::Buffer buf_params;

    vk::DeviceMemory mem_dist;
    vk::DeviceMemory mem_results;
    vk::DeviceMemory mem_changed_0;
    vk::DeviceMemory mem_changed_1;
    vk::DeviceMemory mem_dispatch_deltastep;
    vk::DeviceMemory mem_params;

    uint32_t *gpu_results;

    uint32_t num_nodes;
    uint32_t num_blocks;
    vk::Device &device;
};

} // namespace gpusssp::gpu

#endif
