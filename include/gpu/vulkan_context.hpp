#pragma once

#include "common/logger.hpp"

#include <cstdlib>
#include <vulkan/vulkan.hpp>

namespace gpusssp::gpu
{

namespace detail
{
static constexpr size_t GPUSSSP_DEFAULT_DEVICE = 0;

inline auto selectDevice()
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
    explicit VulkanContext(const char *app_name, uint32_t device_index)
    {
        vk::ApplicationInfo appInfo(app_name, 1, "NoEngine", 1, VK_API_VERSION_1_3);
        m_instance = vk::createInstance({{}, &appInfo});
        auto physDevices = m_instance.enumeratePhysicalDevices();

        if (device_index >= physDevices.size())
        {
            common::log_error() << "Error: device_index " << device_index
                                << " is out of range. Found " << physDevices.size() << " device(s)."
                                << std::endl;
            m_instance.destroy();
            throw std::runtime_error("Invalid device_index value");
        }
        m_physical_device = physDevices[device_index];
        common::log() << "Using device " << device_index << ": "
                      << m_physical_device.getProperties().deviceName << std::endl;

        float queuePriority = 1.0f;
        vk::DeviceQueueCreateInfo queueInfo({}, 0, 1, &queuePriority);
        m_device = m_physical_device.createDevice({{}, 1, &queueInfo});
        m_queue = m_device.getQueue(0, 0);

        m_command_pool =
            m_device.createCommandPool({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, 0});
    }

    ~VulkanContext()
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
    vk::Queue queue() const { return m_queue; }
    vk::CommandPool command_pool() const { return m_command_pool; }
    vk::PhysicalDeviceMemoryProperties memory_properties() const
    {
        return m_physical_device.getMemoryProperties();
    }
    std::string device_name() const { return m_physical_device.getProperties().deviceName; }

  private:
    vk::Instance m_instance;
    vk::PhysicalDevice m_physical_device;
    vk::Device m_device;
    vk::Queue m_queue;
    vk::CommandPool m_command_pool;
};

} // namespace gpusssp::gpu
