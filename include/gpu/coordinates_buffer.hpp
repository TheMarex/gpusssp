#ifndef GPUSSSP_GPU_COORDINATES_BUFFER_HPP
#define GPUSSSP_GPU_COORDINATES_BUFFER_HPP

#include "common/coordinate.hpp"
#include "gpu/memory.hpp"

#include <array>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace gpusssp::gpu
{

class CoordinatesBuffer
{
  public:
    CoordinatesBuffer(const std::vector<common::Coordinate> &coordinates,
                      vk::Device &device,
                      const vk::PhysicalDeviceMemoryProperties &mem_props,
                      vk::CommandPool command_pool,
                      vk::Queue queue)
        : device(device), count(coordinates.size())
    {
        buf_input = gpu::create_exclusive_buffer<int32_t>(
            device,
            coordinates.size() * 2,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);

        buf_output = gpu::create_exclusive_buffer<float>(
            device,
            coordinates.size() * 2,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer);

        mem_input = gpu::alloc_and_bind(
            device, mem_props, buf_input, vk::MemoryPropertyFlagBits::eDeviceLocal);
        mem_output = gpu::alloc_and_bind(
            device, mem_props, buf_output, vk::MemoryPropertyFlagBits::eDeviceLocal);

        std::vector<gpu::BufferCopyInfo> copies = {
            {coordinates.data(), buf_input, coordinates.size() * sizeof(common::Coordinate)}};

        gpu::copy_buffers_batched(device, mem_props, command_pool, queue, copies);
    }

    ~CoordinatesBuffer()
    {
        device.destroyBuffer(buf_input);
        device.destroyBuffer(buf_output);
        device.freeMemory(mem_input);
        device.freeMemory(mem_output);
    }

    std::array<vk::Buffer, 2> buffers() const { return {buf_input, buf_output}; }

    size_t num_vertices() const { return count; }

  private:
    vk::Buffer buf_input;
    vk::Buffer buf_output;
    vk::DeviceMemory mem_input;
    vk::DeviceMemory mem_output;
    vk::Device &device;
    size_t count;
};

} // namespace gpusssp::gpu

#endif
