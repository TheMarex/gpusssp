#ifndef GPUSSSP_GPU_BELLMANFORD_BUFFERS_HPP
#define GPUSSSP_GPU_BELLMANFORD_BUFFERS_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <vulkan/vulkan.hpp>

#include "common/constants.hpp"
#include "gpu/memory.hpp"

namespace gpusssp::gpu
{

class BellmanFordBuffers
{
  public:
    BellmanFordBuffers(size_t num_nodes,
                       vk::Device device,
                       const vk::PhysicalDeviceMemoryProperties &mem_props)
        : num_nodes(static_cast<uint32_t>(num_nodes)), device(device)
    {
        buf_dist = gpu::create_exclusive_buffer<uint32_t>(
            device,
            num_nodes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                vk::BufferUsageFlagBits::eTransferSrc);
        buf_results = gpu::create_exclusive_buffer<uint32_t>(
            device, 2, vk::BufferUsageFlagBits::eTransferDst);
        buf_changed = gpu::create_exclusive_buffer<uint32_t>(
            device,
            1,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                vk::BufferUsageFlagBits::eTransferSrc);

        mem_dist = gpu::alloc_and_bind(
            device, mem_props, buf_dist, vk::MemoryPropertyFlagBits::eDeviceLocal);
        mem_results = gpu::alloc_and_bind(device,
                                          mem_props,
                                          buf_results,
                                          vk::MemoryPropertyFlagBits::eHostVisible |
                                              vk::MemoryPropertyFlagBits::eHostCoherent);
        mem_changed = gpu::alloc_and_bind(
            device, mem_props, buf_changed, vk::MemoryPropertyFlagBits::eDeviceLocal);

        auto *mapped =
            static_cast<uint32_t *>(device.mapMemory(mem_results, 0, 2 * sizeof(uint32_t)));
        gpu_results = mapped;
        gpu_changed = mapped + 1;
    }

    ~BellmanFordBuffers()
    {
        device.unmapMemory(mem_results);
        device.destroyBuffer(buf_dist);
        device.destroyBuffer(buf_results);
        device.destroyBuffer(buf_changed);
        device.freeMemory(mem_dist);
        device.freeMemory(mem_results);
        device.freeMemory(mem_changed);
    }

    uint32_t *best_distance() { return gpu_results; }
    uint32_t *changed() { return gpu_changed; }

    [[nodiscard]] vk::Buffer dist_buffer() const { return buf_dist; }
    [[nodiscard]] vk::Buffer changed_buffer() const { return buf_changed; }

    void cmd_init_dist(vk::CommandBuffer cmd_buf, uint32_t src_node)
    {
        if (src_node > 0)
        {
            cmd_buf.fillBuffer(buf_dist, 0, src_node * sizeof(uint32_t), common::INF_WEIGHT);
        }
        cmd_buf.fillBuffer(buf_dist, src_node * sizeof(uint32_t), sizeof(uint32_t), 0);
        if (src_node < num_nodes - 1)
        {
            cmd_buf.fillBuffer(
                buf_dist, (src_node + 1) * sizeof(uint32_t), VK_WHOLE_SIZE, common::INF_WEIGHT);
        }
    }

    void cmd_clear_changed(vk::CommandBuffer cmd_buf)
    {
        cmd_buf.fillBuffer(buf_changed, 0, VK_WHOLE_SIZE, 0);
    }

    void cmd_sync_dist(vk::CommandBuffer cmd_buf, uint32_t dst_node)
    {
        vk::BufferCopy copy{dst_node * sizeof(uint32_t), 0, sizeof(uint32_t)};
        cmd_buf.copyBuffer(buf_dist, buf_results, 1, &copy);
    }

    void cmd_sync_changed(vk::CommandBuffer cmd_buf)
    {
        vk::BufferCopy copy{0, sizeof(uint32_t), sizeof(uint32_t)};
        cmd_buf.copyBuffer(buf_changed, buf_results, 1, &copy);
    }

  private:
    vk::Buffer buf_dist;
    vk::Buffer buf_results;
    vk::Buffer buf_changed;

    vk::DeviceMemory mem_dist;
    vk::DeviceMemory mem_results;
    vk::DeviceMemory mem_changed;

    uint32_t *gpu_results = nullptr;
    uint32_t *gpu_changed = nullptr;

    uint32_t num_nodes = 0;
    vk::Device device;
};

} // namespace gpusssp::gpu

#endif
