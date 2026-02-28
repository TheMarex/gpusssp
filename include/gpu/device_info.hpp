#ifndef GPUSSSP_GPU_DEVICE_INFO_HPP
#define GPUSSSP_GPU_DEVICE_INFO_HPP

#include "common/logger.hpp"
#include "gpu/vulkan_context.hpp"

#include <vulkan/vulkan.hpp>

namespace gpusssp::gpu
{

inline void print_device_info(const VulkanContext &context)
{
    auto physical_device = context.physical_device();

    auto properties = physical_device.getProperties();

    vk::PhysicalDeviceSubgroupProperties subgroup_props;
    vk::PhysicalDeviceProperties2 properties2;
    properties2.pNext = &subgroup_props;
    physical_device.getProperties2(&properties2);

    auto mem_properties = physical_device.getMemoryProperties();

    uint64_t total_memory = 0;
    for (uint32_t i = 0; i < mem_properties.memoryHeapCount; i++)
    {
        if (mem_properties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal)
        {
            total_memory += mem_properties.memoryHeaps[i].size;
        }
    }

    common::log() << "\n=== Device Information ===" << '\n';
    common::log() << "Device: " << properties.deviceName << '\n';

    uint32_t api_version = properties.apiVersion;
    common::log() << "Max Vulkan Version: " << VK_API_VERSION_MAJOR(api_version) << "."
                  << VK_API_VERSION_MINOR(api_version) << "." << VK_API_VERSION_PATCH(api_version)
                  << '\n';

    common::log() << "Subgroup Size: " << subgroup_props.subgroupSize << " threads" << '\n';

    common::log() << "Max Workgroup Size: " << properties.limits.maxComputeWorkGroupInvocations
                  << " invocations" << '\n';

    common::log() << "Device Memory: " << (total_memory / 1024 / 1024) << " MB" << '\n';
    common::log() << "==========================\n" << '\n';
}

} // namespace gpusssp::gpu

#endif
