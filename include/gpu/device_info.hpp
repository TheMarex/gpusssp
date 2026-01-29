#ifndef GPUSSSP_GPU_DEVICE_INFO_HPP
#define GPUSSSP_GPU_DEVICE_INFO_HPP

#include "gpu/vulkan_context.hpp"

#include <iostream>
#include <vulkan/vulkan.hpp>

namespace gpusssp::gpu
{

inline void printDeviceInfo(const VulkanContext &context)
{
    auto physicalDevice = context.physical_device();

    auto properties = physicalDevice.getProperties();

    vk::PhysicalDeviceSubgroupProperties subgroupProps;
    vk::PhysicalDeviceProperties2 properties2;
    properties2.pNext = &subgroupProps;
    physicalDevice.getProperties2(&properties2);

    auto memProperties = physicalDevice.getMemoryProperties();

    uint64_t totalMemory = 0;
    for (uint32_t i = 0; i < memProperties.memoryHeapCount; i++)
    {
        if (memProperties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal)
        {
            totalMemory += memProperties.memoryHeaps[i].size;
        }
    }

    std::cout << "\n=== Device Information ===" << std::endl;
    std::cout << "Device: " << properties.deviceName << std::endl;

    uint32_t apiVersion = properties.apiVersion;
    std::cout << "Max Vulkan Version: " << VK_API_VERSION_MAJOR(apiVersion) << "."
              << VK_API_VERSION_MINOR(apiVersion) << "." << VK_API_VERSION_PATCH(apiVersion)
              << std::endl;

    std::cout << "Subgroup Size: " << subgroupProps.subgroupSize << " threads" << std::endl;

    std::cout << "Max Workgroup Size: " << properties.limits.maxComputeWorkGroupInvocations
              << " invocations" << std::endl;

    std::cout << "Device Memory: " << (totalMemory / 1024 / 1024) << " MB" << std::endl;
    std::cout << "==========================\n" << std::endl;
}

} // namespace gpusssp::gpu

#endif
