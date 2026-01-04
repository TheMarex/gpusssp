#ifndef GPUSSSP_GPU_DELTASTEP_BUFFERS_HPP
#define GPUSSSP_GPU_DELTASTEP_BUFFERS_HPP

#include <vulkan/vulkan.hpp>

#include "gpu/memory.hpp"

namespace gpusssp::gpu {

class DeltaStepBuffers {
public:
  DeltaStepBuffers(size_t num_nodes, vk::Device& device) : num_nodes(num_nodes), device(device){
  }

  ~DeltaStepBuffers() {
    device.unmapMemory(mem_dist);
    device.unmapMemory(mem_changed);
    device.destroyBuffer(buf_dist);
    device.destroyBuffer(buf_changed);
    device.freeMemory(mem_dist);
    device.freeMemory(mem_changed);
  }

  void initialize(const vk::PhysicalDeviceMemoryProperties& mem_props) {
    buf_dist = gpu::create_exclusive_buffer<uint32_t>(device, num_nodes, vk::BufferUsageFlagBits::eStorageBuffer);
    buf_changed = gpu::create_exclusive_buffer<uint32_t>(device, 1, vk::BufferUsageFlagBits::eStorageBuffer);

    mem_dist = gpu::alloc_and_bind(device, mem_props, buf_dist, vk::MemoryPropertyFlagBits::eHostVisible);
    mem_changed = gpu::alloc_and_bind(device, mem_props, buf_changed, vk::MemoryPropertyFlagBits::eHostVisible);

    gpu_dist = (uint32_t *)device.mapMemory(mem_dist, 0, num_nodes * sizeof(uint32_t));
    gpu_changed = (uint32_t *)device.mapMemory(mem_changed, 0, sizeof(uint32_t));
  }

  uint32_t* dist() {
    return gpu_dist;
  }

  uint32_t* changed() {
    return gpu_changed;
  }

  std::array<const vk::Buffer*, 2> buffers() const {
    return {&buf_dist, &buf_changed};
  }

private:
  vk::Buffer buf_dist;
  vk::Buffer buf_changed;

  vk::DeviceMemory mem_dist;
  vk::DeviceMemory mem_changed;

  uint32_t* gpu_dist;
  uint32_t* gpu_changed;

  size_t num_nodes;
  vk::Device& device;
};

}

#endif
