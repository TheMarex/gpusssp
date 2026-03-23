#ifndef GPUSSSP_GPU_VULKAN_GRAPHICS_CONTEXT_HPP
#define GPUSSSP_GPU_VULKAN_GRAPHICS_CONTEXT_HPP

#include "gpu/vulkan_context.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace gpusssp::gpu
{

class VulkanGraphicsContext : public VulkanContext
{
  public:
    explicit VulkanGraphicsContext(const char *app_name, // NOLINT
                                   uint32_t width,
                                   uint32_t height,
                                   uint32_t device_index)
        : m_width(width), m_height(height)
    {
        initialize_glfw(app_name);

        uint32_t glfw_extension_count = 0;
        const char **glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);
        std::vector<const char *> extensions(glfw_extensions,
                                             glfw_extensions + glfw_extension_count);

        create_instance(app_name, extensions);
        create_vulkan_surface();
        select_physical_device(device_index);
        find_graphics_queue_family();

        std::vector<const char *> device_extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                                                       VK_EXT_ROBUSTNESS_2_EXTENSION_NAME};
        create_device_and_queue(m_graphics_queue_family, device_extensions);
        create_command_pool(m_graphics_queue_family);

        create_swapchain();
        create_render_pass();
        create_framebuffers();
        create_sync_objects();
    }

    ~VulkanGraphicsContext() override
    {
        if (m_device)
        {
            m_device.waitIdle();

            for (auto fence : m_in_flight_fences)
            {
                m_device.destroyFence(fence);
            }
            for (auto semaphore : m_render_finished_semaphores)
            {
                m_device.destroySemaphore(semaphore);
            }
            for (auto semaphore : m_image_available_semaphores)
            {
                m_device.destroySemaphore(semaphore);
            }

            for (auto framebuffer : m_framebuffers)
            {
                m_device.destroyFramebuffer(framebuffer);
            }

            if (m_render_pass)
            {
                m_device.destroyRenderPass(m_render_pass);
            }

            for (auto image_view : m_swapchain_image_views)
            {
                m_device.destroyImageView(image_view);
            }

            if (m_swapchain)
            {
                m_device.destroySwapchainKHR(m_swapchain);
            }
        }

        if (m_surface)
        {
            m_instance.destroySurfaceKHR(m_surface);
        }

        if (m_window)
        {
            glfwDestroyWindow(m_window);
        }

        glfwTerminate();
    }

    VulkanGraphicsContext(const VulkanGraphicsContext &) = delete;
    VulkanGraphicsContext &operator=(const VulkanGraphicsContext &) = delete;
    VulkanGraphicsContext(VulkanGraphicsContext &&) = delete;
    VulkanGraphicsContext &operator=(VulkanGraphicsContext &&) = delete;

    vk::RenderPass render_pass() const { return m_render_pass; }
    vk::Extent2D swapchain_extent() const { return m_swapchain_extent; }
    vk::Format swapchain_format() const { return m_swapchain_image_format; }
    GLFWwindow *window() const { return m_window; }
    uint32_t graphics_queue_family() const { return m_graphics_queue_family; }

    bool should_close() const { return glfwWindowShouldClose(m_window); }

    void poll_events() const { glfwPollEvents(); } // NOLINT

    struct FrameResources
    {
        uint32_t image_index;
        vk::Image image;
        vk::Framebuffer framebuffer;
        vk::Semaphore image_available;
        vk::Semaphore render_finished;
        vk::Fence in_flight;
    };

    FrameResources begin_frame()
    {
        (void)m_device.waitForFences(1, &m_in_flight_fences[m_current_frame], VK_TRUE, UINT64_MAX);

        uint32_t image_index;
        auto result = m_device.acquireNextImageKHR(m_swapchain,
                                                   UINT64_MAX,
                                                   m_image_available_semaphores[m_current_frame],
                                                   nullptr,
                                                   &image_index);

        if (result == vk::Result::eErrorOutOfDateKHR)
        {
            recreate_swapchain();
            return begin_frame();
        }
        else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
        {
            throw std::runtime_error("Failed to acquire swapchain image");
        }

        (void)m_device.resetFences(1, &m_in_flight_fences[m_current_frame]);

        return {.image_index = image_index,
                .image = m_swapchain_images[image_index],
                .framebuffer = m_framebuffers[image_index],
                .image_available = m_image_available_semaphores[m_current_frame],
                .render_finished = m_render_finished_semaphores[m_current_frame],
                .in_flight = m_in_flight_fences[m_current_frame]};
    }

    void end_frame(const FrameResources &frame)
    {
        vk::PresentInfoKHR present_info(
            1, &frame.render_finished, 1, &m_swapchain, &frame.image_index);

        vk::Result result = vk::Result::eSuccess;
        try
        {
            result = shared_queue().presentKHR(present_info);
        }
        catch (const vk::OutOfDateKHRError &)
        {
            result = vk::Result::eErrorOutOfDateKHR;
        }
        catch (const vk::SurfaceLostKHRError &)
        {
            result = vk::Result::eErrorSurfaceLostKHR;
        }

        if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR ||
            result == vk::Result::eErrorSurfaceLostKHR || m_framebuffer_resized)
        {
            m_framebuffer_resized = false;
            recreate_swapchain();
        }

        m_current_frame = (m_current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void set_framebuffer_resized() { m_framebuffer_resized = true; }

  private:
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    GLFWwindow *m_window = nullptr;
    vk::SurfaceKHR m_surface;
    vk::SwapchainKHR m_swapchain;
    vk::RenderPass m_render_pass;

    std::vector<vk::Image> m_swapchain_images;
    std::vector<vk::ImageView> m_swapchain_image_views;
    std::vector<vk::Framebuffer> m_framebuffers;
    vk::Format m_swapchain_image_format;
    vk::Extent2D m_swapchain_extent;

    std::vector<vk::Semaphore> m_image_available_semaphores;
    std::vector<vk::Semaphore> m_render_finished_semaphores;
    std::vector<vk::Fence> m_in_flight_fences;

    uint32_t m_graphics_queue_family{};
    uint32_t m_width;
    uint32_t m_height;
    uint32_t m_current_frame = 0;
    bool m_framebuffer_resized = false;

    void initialize_glfw(const char *app_name)
    {
        if (!glfwInit())
        {
            throw std::runtime_error("Failed to initialize GLFW");
        }

        if (!glfwVulkanSupported())
        {
            glfwTerminate();
            throw std::runtime_error("Vulkan not supported");
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        m_window = glfwCreateWindow(
            static_cast<int>(m_width), static_cast<int>(m_height), app_name, nullptr, nullptr);
        if (!m_window)
        {
            glfwTerminate();
            throw std::runtime_error("Failed to create GLFW window");
        }
    }

    void create_vulkan_surface()
    {
        VkSurfaceKHR tmp_surface;
        if (glfwCreateWindowSurface(m_instance, m_window, nullptr, &tmp_surface) != VK_SUCCESS)
        {
            m_instance.destroy();
            glfwDestroyWindow(m_window);
            glfwTerminate();
            throw std::runtime_error("Failed to create window surface");
        }
        m_surface = tmp_surface;
    }

    void find_graphics_queue_family()
    {
        auto queue_families = m_physical_device.getQueueFamilyProperties();
        m_graphics_queue_family = UINT32_MAX;

        for (uint32_t i = 0; i < queue_families.size(); ++i)
        {
            if (queue_families[i].queueFlags & vk::QueueFlagBits::eGraphics)
            {
                if (m_physical_device.getSurfaceSupportKHR(i, m_surface))
                {
                    m_graphics_queue_family = i;
                    break;
                }
            }
        }

        if (m_graphics_queue_family == UINT32_MAX)
        {
            m_instance.destroySurfaceKHR(m_surface);
            m_instance.destroy();
            glfwDestroyWindow(m_window);
            glfwTerminate();
            throw std::runtime_error("No suitable queue family found");
        }
    }

    void create_swapchain()
    {
        auto capabilities = m_physical_device.getSurfaceCapabilitiesKHR(m_surface);
        auto formats = m_physical_device.getSurfaceFormatsKHR(m_surface);
        auto present_modes = m_physical_device.getSurfacePresentModesKHR(m_surface);

        vk::SurfaceFormatKHR surface_format = formats[0];
        for (const auto &format : formats)
        {
            if (format.format == vk::Format::eB8G8R8A8Srgb &&
                format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
            {
                surface_format = format;
                break;
            }
        }

        vk::PresentModeKHR present_mode = vk::PresentModeKHR::eFifo;
        for (const auto &mode : present_modes)
        {
            if (mode == vk::PresentModeKHR::eMailbox)
            {
                present_mode = mode;
                break;
            }
        }

        int width;
        int height;
        glfwGetFramebufferSize(m_window, &width, &height);

        vk::Extent2D extent;
        if (capabilities.currentExtent.width != UINT32_MAX)
        {
            extent = capabilities.currentExtent;
        }
        else
        {
            extent = vk::Extent2D(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
            extent.width = std::clamp(
                extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            extent.height = std::clamp(extent.height,
                                       capabilities.minImageExtent.height,
                                       capabilities.maxImageExtent.height);
        }

        uint32_t image_count = std::min(capabilities.minImageCount + 1, capabilities.maxImageCount);

        vk::SwapchainCreateInfoKHR create_info({},
                                               m_surface,
                                               image_count,
                                               surface_format.format,
                                               surface_format.colorSpace,
                                               extent,
                                               1,
                                               vk::ImageUsageFlagBits::eColorAttachment,
                                               vk::SharingMode::eExclusive,
                                               0,
                                               nullptr,
                                               capabilities.currentTransform,
                                               vk::CompositeAlphaFlagBitsKHR::eOpaque,
                                               present_mode,
                                               VK_TRUE);

        m_swapchain = m_device.createSwapchainKHR(create_info);
        m_swapchain_images = m_device.getSwapchainImagesKHR(m_swapchain);
        m_swapchain_image_format = surface_format.format;
        m_swapchain_extent = extent;

        m_swapchain_image_views.resize(m_swapchain_images.size());
        for (size_t i = 0; i < m_swapchain_images.size(); ++i)
        {
            vk::ImageViewCreateInfo view_info({},
                                              m_swapchain_images[i],
                                              vk::ImageViewType::e2D,
                                              m_swapchain_image_format,
                                              {},
                                              {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});

            m_swapchain_image_views[i] = m_device.createImageView(view_info);
        }
    }

    void create_render_pass()
    {
        vk::AttachmentDescription color_attachment({},
                                                   m_swapchain_image_format,
                                                   vk::SampleCountFlagBits::e1,
                                                   vk::AttachmentLoadOp::eClear,
                                                   vk::AttachmentStoreOp::eStore,
                                                   vk::AttachmentLoadOp::eDontCare,
                                                   vk::AttachmentStoreOp::eDontCare,
                                                   vk::ImageLayout::eUndefined,
                                                   vk::ImageLayout::ePresentSrcKHR);

        vk::AttachmentReference color_attachment_ref(0, vk::ImageLayout::eColorAttachmentOptimal);

        vk::SubpassDescription subpass(
            {}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment_ref);

        vk::SubpassDependency dependency(VK_SUBPASS_EXTERNAL,
                                         0,
                                         vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                         vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                         {},
                                         vk::AccessFlagBits::eColorAttachmentWrite);

        vk::RenderPassCreateInfo render_pass_info(
            {}, 1, &color_attachment, 1, &subpass, 1, &dependency);

        m_render_pass = m_device.createRenderPass(render_pass_info);
    }

    void create_framebuffers()
    {
        m_framebuffers.resize(m_swapchain_image_views.size());

        for (size_t i = 0; i < m_swapchain_image_views.size(); ++i)
        {
            vk::ImageView attachments[] = {m_swapchain_image_views[i]};

            vk::FramebufferCreateInfo framebuffer_info({},
                                                       m_render_pass,
                                                       1,
                                                       attachments,
                                                       m_swapchain_extent.width,
                                                       m_swapchain_extent.height,
                                                       1);

            m_framebuffers[i] = m_device.createFramebuffer(framebuffer_info);
        }
    }

    void create_sync_objects()
    {
        m_image_available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
        m_render_finished_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
        m_in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);

        vk::SemaphoreCreateInfo semaphore_info;
        vk::FenceCreateInfo fence_info(vk::FenceCreateFlagBits::eSignaled);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            m_image_available_semaphores[i] = m_device.createSemaphore(semaphore_info);
            m_render_finished_semaphores[i] = m_device.createSemaphore(semaphore_info);
            m_in_flight_fences[i] = m_device.createFence(fence_info);
        }
    }

    void cleanup_swapchain()
    {
        for (auto framebuffer : m_framebuffers)
        {
            m_device.destroyFramebuffer(framebuffer);
        }

        for (auto image_view : m_swapchain_image_views)
        {
            m_device.destroyImageView(image_view);
        }

        m_device.destroySwapchainKHR(m_swapchain);
    }

    void recreate_swapchain()
    {
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(m_window, &width, &height);
        while (width == 0 || height == 0)
        {
            glfwGetFramebufferSize(m_window, &width, &height);
            glfwWaitEvents();
        }

        m_device.waitIdle();

        cleanup_swapchain();

        create_swapchain();
        create_framebuffers();
    }
};

} // namespace gpusssp::gpu

#endif
