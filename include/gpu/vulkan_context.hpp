#pragma once

#include "common/logger.hpp"
#include "gpu/shared_queue.hpp"

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>

namespace gpusssp::gpu
{

namespace detail
{
static constexpr size_t GPUSSSP_DEFAULT_DEVICE = 0;

inline auto select_device()
{
    if (const char *env_device = std::getenv("GPUSSSP_DEVICE"))
    {
        return std::stoul(env_device);
    }

    return GPUSSSP_DEFAULT_DEVICE;
}
} // namespace detail

class VulkanContext
{
  public:
    // Creates a compute-only context by default
    explicit VulkanContext(const char *app_name, uint32_t device_index) : m_shared_queue(nullptr)
    {
        create_instance(app_name, {});
        select_physical_device(device_index);
        create_device_and_queue(0, {});
        create_command_pool(0);
    }

    virtual ~VulkanContext()
    {
        if (m_device)
        {
            if (m_command_pool)
            {
                m_device.destroyCommandPool(m_command_pool);
            }
            m_device.destroy();
        }
        if (m_instance)
        {
            m_instance.destroy();
        }
    }

    VulkanContext(const VulkanContext &) = delete;
    VulkanContext &operator=(const VulkanContext &) = delete;
    VulkanContext(VulkanContext &&) = delete;
    VulkanContext &operator=(VulkanContext &&) = delete;

    vk::Instance instance() const { return m_instance; }
    vk::PhysicalDevice physical_device() const { return m_physical_device; }
    vk::Device device() const { return m_device; }
    vk::Queue queue() const { return m_shared_queue.unwrap(); }
    SharedQueue &shared_queue() const { return m_shared_queue; }
    vk::CommandPool command_pool() const { return m_command_pool; }
    vk::PhysicalDeviceMemoryProperties memory_properties() const
    {
        return m_physical_device.getMemoryProperties();
    }
    std::string device_name() const { return m_physical_device.getProperties().deviceName; }

  protected:
    VulkanContext() : m_shared_queue(nullptr) {}

    void create_instance(const char *app_name, const std::vector<const char *> &extensions)
    {
        vk::ApplicationInfo app_info(app_name, 1, "NoEngine", 1, VK_API_VERSION_1_3);
        vk::InstanceCreateInfo instance_info(
            {}, &app_info, 0, nullptr, static_cast<uint32_t>(extensions.size()), extensions.data());
        m_instance = vk::createInstance(instance_info);
    }

    void select_physical_device(uint32_t device_index)
    {
        auto phys_devices = m_instance.enumeratePhysicalDevices();

        if (device_index >= phys_devices.size())
        {
            common::log_error() << "Error: device_index " << device_index
                                << " is out of range. Found " << phys_devices.size()
                                << " device(s)." << '\n';
            m_instance.destroy();
            throw std::runtime_error("Invalid device_index value");
        }
        m_physical_device = phys_devices[device_index];
        common::log() << "Using device " << device_index << ": "
                      << m_physical_device.getProperties().deviceName << '\n';
    }

    void create_device_and_queue(uint32_t queue_family_index,
                                 const std::vector<const char *> &device_extensions)
    {
        float queue_priority = 1.0f;
        vk::DeviceQueueCreateInfo queue_info({}, queue_family_index, 1, &queue_priority);

        vk::PhysicalDeviceFeatures device_features;
        vk::DeviceCreateInfo device_info({},
                                         1,
                                         &queue_info,
                                         0,
                                         nullptr,
                                         device_extensions.size(),
                                         device_extensions.data(),
                                         &device_features);

        m_device = m_physical_device.createDevice(device_info);
        vk::Queue raw_queue = m_device.getQueue(queue_family_index, 0);
        new (&m_shared_queue) SharedQueue(raw_queue);
    }

    void create_command_pool(uint32_t queue_family_index)
    {
        m_command_pool = m_device.createCommandPool(
            {vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queue_family_index});
    }

    vk::Instance m_instance;
    vk::PhysicalDevice m_physical_device;
    vk::Device m_device;
    mutable SharedQueue m_shared_queue;
    vk::CommandPool m_command_pool;
};

} // namespace gpusssp::gpu
