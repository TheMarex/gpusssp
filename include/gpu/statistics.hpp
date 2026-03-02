#ifndef GPUSSSP_GPU_STATISTICS_HPP
#define GPUSSSP_GPU_STATISTICS_HPP

#ifdef ENABLE_STATISTICS
#include <sstream>
#endif

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vulkan/vulkan.hpp>

#include "gpu/memory.hpp"

namespace gpusssp::gpu
{

enum class StatisticsEvent : uint8_t
{
    NEARFAR_RELAX_EDGES,
    NEARFAR_RELAX_IMPROVED,
    NEARFAR_COMPACT_NODES,
    DELTASTEP_EDGES_RELAXED,
    DELTASTEP_EDGES_IMPROVED,
    BELLMANFORD_EDGES_RELAXED,
    BELLMANFORD_EDGES_IMPROVED,
    NUM_EVENTS
};

inline const char *event_to_name(StatisticsEvent event)
{
    static std::array<const char *, static_cast<std::size_t>(StatisticsEvent::NUM_EVENTS)> names{
        {"NEARFAR_RELAX_EDGES",
         "NEARFAR_RELAX_IMPROVED",
         "NEARFAR_COMPACT_NODES",
         "DELTASTEP_EDGES_RELAXED",
         "DELTASTEP_EDGES_IMPROVED",
         "BELLMANFORD_EDGES_RELAXED",
         "BELLMANFORD_EDGES_IMPROVED"}};
    return names[static_cast<std::size_t>(event)];
}

class Statistics
{
  public:
    Statistics(vk::Device &device, const vk::PhysicalDeviceMemoryProperties &mem_props)
        : device(device)
    {
#ifdef ENABLE_STATISTICS
        buf_statistics = create_exclusive_buffer<uint64_t>(
            device, NUM_COUNTERS, vk::BufferUsageFlagBits::eStorageBuffer);
        mem_statistics = alloc_and_bind(device,
                                        mem_props,
                                        buf_statistics,
                                        vk::MemoryPropertyFlagBits::eHostVisible |
                                            vk::MemoryPropertyFlagBits::eHostCoherent);
        gpu_statistics_counters =
            (uint64_t *)device.mapMemory(mem_statistics, 0, NUM_COUNTERS * sizeof(uint64_t));
        reset();
#else
        buf_statistics =
            create_exclusive_buffer<uint64_t>(device, 1, vk::BufferUsageFlagBits::eStorageBuffer);
        mem_statistics = alloc_and_bind(
            device, mem_props, buf_statistics, vk::MemoryPropertyFlagBits::eDeviceLocal);
#endif
    }

    ~Statistics()
    {
#ifdef ENABLE_STATISTICS
        device.unmapMemory(mem_statistics);
#endif
        device.destroyBuffer(buf_statistics);
        device.freeMemory(mem_statistics);
    }

    [[nodiscard]] vk::Buffer buffer() const { return buf_statistics; }

    void reset()
    {
#ifdef ENABLE_STATISTICS
        std::memset(gpu_statistics_counters, 0, NUM_COUNTERS * sizeof(uint64_t));
#endif
    }

    [[nodiscard]] std::string summary() // NOLINT
    {
#ifdef ENABLE_STATISTICS
        std::stringstream ss;
        for (size_t i = 0; i < NUM_COUNTERS; ++i)
        {
            StatisticsEvent event = static_cast<StatisticsEvent>(i);
            ss << event_to_name(event) << ": " << gpu_statistics_counters[i] << std::endl;
        }
        return ss.str();
#else
        return "";
#endif
    }

  private:
    static constexpr size_t NUM_COUNTERS = static_cast<size_t>(StatisticsEvent::NUM_EVENTS);

    vk::Device &device;

    vk::Buffer buf_statistics;
    vk::DeviceMemory mem_statistics;

    uint64_t *gpu_statistics_counters{nullptr};
};

} // namespace gpusssp::gpu

#endif
