#pragma once

#include "common/logger.hpp"

#include <cstdlib>
#include <vector>
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
    // Creates a compute-only context by default
    explicit VulkanContext(const char *app_name, uint32_t device_index)
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
    vk::Queue queue() const { return m_queue; }
    vk::CommandPool command_pool() const { return m_command_pool; }
    vk::PhysicalDeviceMemoryProperties memory_properties() const
    {
        return m_physical_device.getMemoryProperties();
    }
    std::string device_name() const { return m_physical_device.getProperties().deviceName; }

  protected:
    VulkanContext() = default;

    void create_instance(const char *app_name, const std::vector<const char *> &extensions)
    {
        vk::ApplicationInfo appInfo(app_name, 1, "NoEngine", 1, VK_API_VERSION_1_3);
        vk::InstanceCreateInfo instanceInfo(
            {}, &appInfo, 0, nullptr, static_cast<uint32_t>(extensions.size()), extensions.data());
        m_instance = vk::createInstance(instanceInfo);
    }

    void select_physical_device(uint32_t device_index)
    {
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
    }

    void create_device_and_queue(uint32_t queue_family_index,
                                 const std::vector<const char *> &device_extensions)
    {
        float queuePriority = 1.0f;
        vk::DeviceQueueCreateInfo queueInfo({}, queue_family_index, 1, &queuePriority);

        vk::PhysicalDeviceFeatures deviceFeatures;
        vk::DeviceCreateInfo deviceInfo({},
                                        1,
                                        &queueInfo,
                                        0,
                                        nullptr,
                                        device_extensions.size(),
                                        device_extensions.data(),
                                        &deviceFeatures);

        m_device = m_physical_device.createDevice(deviceInfo);
        m_queue = m_device.getQueue(queue_family_index, 0);
    }

    void create_command_pool(uint32_t queue_family_index)
    {
        m_command_pool = m_device.createCommandPool(
            {vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queue_family_index});
    }

    vk::Instance m_instance;
    vk::PhysicalDevice m_physical_device;
    vk::Device m_device;
    vk::Queue m_queue;
    vk::CommandPool m_command_pool;
};

} // namespace gpusssp::gpu
