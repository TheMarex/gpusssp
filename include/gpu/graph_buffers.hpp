#ifndef GPUSSSP_GPU_GRAPH_BUFFERS_HPP
#define GPUSSSP_GPU_GRAPH_BUFFERS_HPP

#include <array>

#include <vulkan/vulkan.hpp>

#include "gpu/memory.hpp"

namespace gpusssp::gpu {

template<typename GraphT>
class GraphBuffers {
public:
  GraphBuffers(const GraphT& graph,
      vk::Device& device)
    : graph(graph), device(device) {
  }

  ~GraphBuffers() {
    device.destroyBuffer(buf_first_edges);
    device.destroyBuffer(buf_targets);
    device.destroyBuffer(buf_weights);
    device.freeMemory(mem_first_edges);
    device.freeMemory(mem_targets);
    device.freeMemory(mem_weights);
  }

  void initialize(const vk::PhysicalDeviceMemoryProperties& mem_props) {
    buf_first_edges =
        gpu::create_exclusive_buffer<uint32_t>(device, graph.first_edges.size(), vk::BufferUsageFlagBits::eStorageBuffer);
    buf_targets =
        gpu::create_exclusive_buffer<uint32_t>(device, graph.targets.size(), vk::BufferUsageFlagBits::eStorageBuffer);
    buf_weights =
        gpu::create_exclusive_buffer<uint32_t>(device, graph.weights.size(), vk::BufferUsageFlagBits::eStorageBuffer);

    mem_first_edges = gpu::alloc_and_bind(device, mem_props, buf_first_edges, vk::MemoryPropertyFlagBits::eHostVisible);
    mem_targets = gpu::alloc_and_bind(device, mem_props, buf_targets, vk::MemoryPropertyFlagBits::eHostVisible);
    mem_weights = gpu::alloc_and_bind(device, mem_props, buf_weights, vk::MemoryPropertyFlagBits::eHostVisible);

    memcpy(device.mapMemory(mem_first_edges, 0, graph.first_edges.size() * sizeof(uint32_t)),
           graph.first_edges.data(),
           graph.first_edges.size() * sizeof(uint32_t));
    memcpy(device.mapMemory(mem_targets, 0, graph.targets.size() * sizeof(uint32_t)),
           graph.targets.data(),
           graph.targets.size() * sizeof(uint32_t));
    memcpy(device.mapMemory(mem_weights, 0, graph.weights.size() * sizeof(uint32_t)),
           graph.weights.data(),
           graph.weights.size() * sizeof(uint32_t));
    device.unmapMemory(mem_first_edges);
    device.unmapMemory(mem_targets);
    device.unmapMemory(mem_weights);
  }

  auto num_nodes() const {
    return graph.num_nodes();
  }

  std::array<vk::Buffer, 3> buffers() const {
    return {buf_first_edges, buf_targets, buf_weights};
  }

private:
  vk::Buffer buf_first_edges;
  vk::Buffer buf_targets;
  vk::Buffer buf_weights;

  vk::DeviceMemory mem_first_edges;
  vk::DeviceMemory mem_targets;
  vk::DeviceMemory mem_weights; 

  const GraphT& graph;
  vk::Device& device;
};

}

#endif
