#ifndef GPUSSSP_MEMORY_HPP
#define GPUSSSP_MEMORY_HPP

#include <vulkan/vulkan.hpp>

namespace gpusssp::gpu {
  inline auto find_memory_type_index(const vk::PhysicalDeviceMemoryProperties& mem_props, const vk::MemoryRequirements& mr, vk::MemoryPropertyFlagBits flags) {
      for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++)
      {
          if ((mr.memoryTypeBits & (1 << i)) &&
              (mem_props.memoryTypes[i].propertyFlags & flags))
          {
              return i;
          }
      }

      return 0u;
  }

  inline auto alloc_and_bind(vk::Device& device, const vk::PhysicalDeviceMemoryProperties& mem_props, const vk::Buffer& buf, vk::MemoryPropertyFlagBits flags)
  {
      vk::MemoryRequirements mr = device.getBufferMemoryRequirements(buf);
      auto mem_type_index = find_memory_type_index(mem_props, mr, flags);
      vk::DeviceMemory mem = device.allocateMemory({mr.size, mem_type_index});
      device.bindBufferMemory(buf, mem, 0);
      return mem;
  }

  template <typename T>
  auto create_exclusive_buffer(vk::Device& device, size_t size, vk::BufferUsageFlags usage) {
      return device.createBuffer({{}, sizeof(T) * size, usage, vk::SharingMode::eExclusive});
  }
}

#endif
