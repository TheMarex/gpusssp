#ifndef GPUSSSP_GPU_GRAPH_BUFFERS_HPP
#define GPUSSSP_GPU_GRAPH_BUFFERS_HPP

#include <array>

#include <vulkan/vulkan.hpp>

#include "gpu/memory.hpp"

namespace gpusssp::gpu
{

template <typename GraphT> class GraphBuffers
{
  public:
    GraphBuffers(const GraphT &graph,
                 vk::Device &device,
                 const vk::PhysicalDeviceMemoryProperties &mem_props,
                 vk::CommandPool command_pool,
                 vk::Queue queue)
        : graph(graph), device(device)
    {
        // Create device-local buffers with transfer destination flag
        buf_first_edges = gpu::create_exclusive_buffer<uint32_t>(
            device,
            graph.first_edges.size(),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);
        buf_targets = gpu::create_exclusive_buffer<uint32_t>(
            device,
            graph.targets.size(),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);
        buf_weights = gpu::create_exclusive_buffer<uint32_t>(
            device,
            graph.weights.size(),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);

        // Allocate device-local memory
        mem_first_edges = gpu::alloc_and_bind(
            device, mem_props, buf_first_edges, vk::MemoryPropertyFlagBits::eDeviceLocal);
        mem_targets = gpu::alloc_and_bind(
            device, mem_props, buf_targets, vk::MemoryPropertyFlagBits::eDeviceLocal);
        mem_weights = gpu::alloc_and_bind(
            device, mem_props, buf_weights, vk::MemoryPropertyFlagBits::eDeviceLocal);

        // Prepare batch copy operations
        std::vector<gpu::BufferCopyInfo> copies = {
            {graph.first_edges.data(),
             buf_first_edges,
             graph.first_edges.size() * sizeof(uint32_t)},
            {graph.targets.data(), buf_targets, graph.targets.size() * sizeof(uint32_t)},
            {graph.weights.data(), buf_weights, graph.weights.size() * sizeof(uint32_t)}};

        // Perform batched copy from host to device-local memory
        gpu::copy_buffers_batched(device, mem_props, command_pool, queue, copies);
    }

    ~GraphBuffers()
    {
        device.destroyBuffer(buf_first_edges);
        device.destroyBuffer(buf_targets);
        device.destroyBuffer(buf_weights);
        device.freeMemory(mem_first_edges);
        device.freeMemory(mem_targets);
        device.freeMemory(mem_weights);
    }

    auto num_nodes() const { return graph.num_nodes(); }
    auto num_edges() const { return graph.num_edges(); }

    std::array<vk::Buffer, 3> buffers() const
    {
        return {buf_first_edges, buf_targets, buf_weights};
    }

  private:
    vk::Buffer buf_first_edges;
    vk::Buffer buf_targets;
    vk::Buffer buf_weights;

    vk::DeviceMemory mem_first_edges;
    vk::DeviceMemory mem_targets;
    vk::DeviceMemory mem_weights;

    const GraphT &graph;
    vk::Device &device;
};

} // namespace gpusssp::gpu

#endif
