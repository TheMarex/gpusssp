#ifndef GPUSSSP_TESTS_VULKAN_TEST_FIXTURE_HPP
#define GPUSSSP_TESTS_VULKAN_TEST_FIXTURE_HPP

#include "gpu/vulkan_context.hpp"

#include <stdexcept>
#include <vulkan/vulkan.hpp>

namespace gpusssp::test
{

/**
 * RAII wrapper for Vulkan resources needed for testing.
 * Automatically initializes Vulkan instance, device, queue, and command pool.
 * Cleans up all resources in destructor.
 */
class VulkanTestFixture
{
  public:
    VulkanTestFixture() { initialize(); }

    ~VulkanTestFixture() { cleanup(); }

    // Delete copy/move constructors to ensure RAII semantics
    VulkanTestFixture(const VulkanTestFixture &) = delete;
    VulkanTestFixture &operator=(const VulkanTestFixture &) = delete;
    VulkanTestFixture(VulkanTestFixture &&) = delete;
    VulkanTestFixture &operator=(VulkanTestFixture &&) = delete;

    // Accessors
    vk::Instance get_instance() const { return instance; }
    vk::PhysicalDevice get_physical_device() const { return physical_device; }
    vk::Device get_device() const { return device; }
    vk::Queue get_queue() const { return queue; }
    vk::CommandPool get_command_pool() const { return command_pool; }
    vk::PhysicalDeviceMemoryProperties get_memory_properties() const { return memory_properties; }

  private:
    void initialize()
    {
        // Create Vulkan instance
        vk::ApplicationInfo app_info("DeltaStep Test", 1, "NoEngine", 1, VK_API_VERSION_1_2);

        instance = vk::createInstance({{}, &app_info});

        // Select physical device (use first available)
        auto physical_devices = instance.enumeratePhysicalDevices();
        if (physical_devices.empty())
        {
            throw std::runtime_error("No Vulkan-capable devices found");
        }
        auto device_index = gpu::detail::selectDevice();
        physical_device = physical_devices[device_index];

        // Get memory properties for later use
        memory_properties = physical_device.getMemoryProperties();

        // Create logical device with compute queue
        float queue_priority = 1.0f;
        vk::DeviceQueueCreateInfo queue_info({}, 0, 1, &queue_priority);
        device = physical_device.createDevice({{}, 1, &queue_info});

        // Get queue handle
        queue = device.getQueue(0, 0);

        // Create command pool
        command_pool =
            device.createCommandPool({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, 0});
    }

    void cleanup()
    {
        if (device)
        {
            if (command_pool)
            {
                device.destroyCommandPool(command_pool);
            }
            device.destroy();
        }
        if (instance)
        {
            instance.destroy();
        }
    }

    vk::Instance instance;
    vk::PhysicalDevice physical_device;
    vk::Device device;
    vk::Queue queue;
    vk::CommandPool command_pool;
    vk::PhysicalDeviceMemoryProperties memory_properties;
};

} // namespace gpusssp::test

#endif // GPUSSSP_TESTS_VULKAN_TEST_FIXTURE_HPP
