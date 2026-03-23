#include "common/cli.hpp"
#include "common/coordinate.hpp"
#include "common/files.hpp"
#include "common/graph_metrics.hpp"
#include "common/logger.hpp"
#include "common/nearest_neighbour.hpp"
#include "common/web_mercator.hpp"
#include "common/weighted_graph.hpp"
#include "gpu/coordinates_buffer.hpp"
#include "gpu/shader.hpp"
#include "gpu/shared_queue.hpp"
#include "gpu/vulkan_graphics_context.hpp"
#include <argparse/argparse.hpp>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vulkan/vulkan_core.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <GLFW/glfw3.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <latch>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <thread>
#include <vector>
#include <vulkan/vulkan.hpp>

#include "gpu/deltastep.hpp"
#include "gpu/deltastep_buffers.hpp"
#include "gpu/graph_buffers.hpp"
#include "gpu/nearfar.hpp"
#include "gpu/nearfar_buffers.hpp"
#include "gpu/statistics.hpp"
#include <variant>

using namespace gpusssp; // NOLINT

using DeltaStepPayload = gpu::DeltaStepPayload;
using NearFarPayload = gpu::NearFarPayload;
using DeltaStepTracer = gpu::DeltaStepTracer;
using NearFarTracer = gpu::NearFarTracer;

using VisualizationPayload = std::variant<DeltaStepPayload, NearFarPayload>;

enum class AlgorithmType : uint8_t
{
    DELTA_STEP,
    NEAR_FAR
};

struct App;

class AlgorithmRunner
{
  public:
    virtual ~AlgorithmRunner() = default;
    virtual void initialize(vk::CommandPool cmd_pool) = 0;
    virtual uint32_t run_with_trace(vk::CommandPool cmd_pool,
                                    gpu::SharedQueue &queue,
                                    uint32_t src_node,
                                    uint32_t dst_node,
                                    App &app) = 0;
    virtual std::optional<VisualizationPayload> payload(App &app) const = 0;
    virtual AlgorithmType type() const = 0;
    virtual vk::Buffer dist_buffer() const = 0;
    virtual std::array<vk::Buffer, 2> changed_buffers() const = 0;
    virtual std::array<vk::Buffer, 2> near_buffers() const = 0;
    virtual std::array<vk::Buffer, 2> far_buffers() const = 0;
    virtual uint32_t max_distance() const = 0;
    virtual uint32_t delta() const = 0;
};

class FrameRecorder
{
  public:
    FrameRecorder(gpu::VulkanGraphicsContext &context, std::string output_base_dir)
        : m_context(context), m_output_base_dir(std::move(output_base_dir))
    {
    }

    ~FrameRecorder()
    {
        if (m_is_recording)
        {
            stop_recording();
        }
    }

    FrameRecorder(const FrameRecorder &) = delete;
    FrameRecorder &operator=(const FrameRecorder &) = delete;
    FrameRecorder(FrameRecorder &&) = delete;
    FrameRecorder &operator=(FrameRecorder &&) = delete;

    void start_recording(vk::Extent2D extent)
    {
        if (m_is_recording)
        {
            return;
        }

        auto now = std::chrono::system_clock::now();
        auto timestamp =
            std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();

        m_current_recording_dir = m_output_base_dir + "/" + std::to_string(timestamp);
        std::filesystem::create_directories(m_current_recording_dir);

        m_frame_number = 0;
        m_is_recording = true;
        allocate_staging_buffer(extent);

        common::log() << "Recording started: " << m_current_recording_dir << '\n';
    }

    void stop_recording()
    {
        if (!m_is_recording)
        {
            return;
        }

        m_is_recording = false;
        free_staging_buffer();

        common::log() << "Recording stopped. " << m_frame_number << " frames captured." << '\n';
        common::log() << "To convert to GIF, run:" << '\n';
        common::log() << "  ffmpeg -framerate 60 -i " << m_current_recording_dir
                      << "/frame_%06d.png -vf \"fps=30\" " << m_current_recording_dir
                      << "/output.gif" << '\n';
    }

    void capture_frame(vk::Image swapchain_image, vk::Extent2D extent)
    {
        if (!m_is_recording)
        {
            return;
        }

        if (extent.width * extent.height * 4 != m_buffer_size)
        {
            free_staging_buffer();
            allocate_staging_buffer(extent);
        }

        copy_image_to_buffer(swapchain_image, extent);
        save_buffer_as_png(extent);
        m_frame_number++;
    }

    [[nodiscard]] bool is_recording() const { return m_is_recording; }

  private:
    gpu::VulkanGraphicsContext &m_context;

    std::string m_output_base_dir;
    std::string m_current_recording_dir;
    uint32_t m_frame_number = 0;
    bool m_is_recording = false;

    vk::Buffer m_staging_buffer;
    vk::DeviceMemory m_staging_memory;
    size_t m_buffer_size = 0;

    void allocate_staging_buffer(vk::Extent2D extent)
    {
        m_buffer_size = static_cast<size_t>(extent.width) * extent.height * 4;

        m_staging_buffer = gpu::create_exclusive_buffer<uint8_t>(
            m_context.device(), m_buffer_size, vk::BufferUsageFlagBits::eTransferDst);

        m_staging_memory = gpu::alloc_and_bind(m_context.device(),
                                               m_context.memory_properties(),
                                               m_staging_buffer,
                                               vk::MemoryPropertyFlagBits::eHostVisible |
                                                   vk::MemoryPropertyFlagBits::eHostCoherent);
    }

    void free_staging_buffer()
    {
        if (m_staging_buffer)
        {
            m_context.device().destroyBuffer(m_staging_buffer);
            m_staging_buffer = nullptr;
        }
        if (m_staging_memory)
        {
            m_context.device().freeMemory(m_staging_memory);
            m_staging_memory = nullptr;
        }
        m_buffer_size = 0;
    }

    void copy_image_to_buffer(vk::Image image, vk::Extent2D extent)
    {
        vk::CommandBuffer cmd = m_context.device().allocateCommandBuffers(
            {m_context.command_pool(), vk::CommandBufferLevel::ePrimary, 1})[0];

        cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        vk::ImageMemoryBarrier barrier_to_transfer(vk::AccessFlagBits::eMemoryRead,
                                                   vk::AccessFlagBits::eTransferRead,
                                                   vk::ImageLayout::ePresentSrcKHR,
                                                   vk::ImageLayout::eTransferSrcOptimal,
                                                   VK_QUEUE_FAMILY_IGNORED,
                                                   VK_QUEUE_FAMILY_IGNORED,
                                                   image,
                                                   {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});

        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eTransfer,
                            {},
                            {},
                            {},
                            barrier_to_transfer);

        vk::BufferImageCopy region(0,
                                   0,
                                   0,
                                   {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                                   {0, 0, 0},
                                   {extent.width, extent.height, 1});

        cmd.copyImageToBuffer(
            image, vk::ImageLayout::eTransferSrcOptimal, m_staging_buffer, 1, &region);

        vk::ImageMemoryBarrier barrier_to_present(vk::AccessFlagBits::eTransferRead,
                                                  vk::AccessFlagBits::eMemoryRead,
                                                  vk::ImageLayout::eTransferSrcOptimal,
                                                  vk::ImageLayout::ePresentSrcKHR,
                                                  VK_QUEUE_FAMILY_IGNORED,
                                                  VK_QUEUE_FAMILY_IGNORED,
                                                  image,
                                                  {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});

        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eBottomOfPipe,
                            {},
                            {},
                            {},
                            barrier_to_present);

        cmd.end();

        vk::Fence fence = m_context.device().createFence({});
        m_context.shared_queue().submit({{0, nullptr, nullptr, 1, &cmd}}, fence);
        (void)m_context.device().waitForFences(1, &fence, VK_TRUE, UINT64_MAX);

        m_context.device().destroyFence(fence);
        m_context.device().freeCommandBuffers(m_context.command_pool(), 1, &cmd);
    }

    void save_buffer_as_png(vk::Extent2D extent)
    {
        std::ostringstream filename;
        filename << m_current_recording_dir << "/frame_" << std::setw(6) << std::setfill('0')
                 << m_frame_number << ".png";

        uint32_t width = extent.width;
        uint32_t height = extent.height;

        auto *data = static_cast<uint8_t *>(
            m_context.device().mapMemory(m_staging_memory, 0, m_buffer_size));

        // Convert from BGRA (Vulkan) to RGBA (PNG) and write
        std::vector<uint8_t> rgba_data(width * height * 4);
        for (uint32_t i = 0; i < width * height; ++i)
        {
            rgba_data[(i * 4) + 0] = data[(i * 4) + 2];
            rgba_data[(i * 4) + 1] = data[(i * 4) + 1];
            rgba_data[(i * 4) + 2] = data[(i * 4) + 0];
            rgba_data[(i * 4) + 3] = data[(i * 4) + 3];
        }

        m_context.device().unmapMemory(m_staging_memory);

        int stride = static_cast<int>(width * 4);
        int result = stbi_write_png(filename.str().c_str(),
                                    static_cast<int>(width),
                                    static_cast<int>(height),
                                    4,
                                    rgba_data.data(),
                                    stride);

        if (result == 0)
        {
            common::log_error() << "Failed to write PNG file: " << filename.str() << '\n';
        }
    }
};

enum class ColorMode : uint8_t
{
    FIXED = 0,
    ORDERING = 1,
    WORKGROUP = 2,
    TRACE = 8,
    TRACE_DISTANCE = 9,
    TRACE_BUCKET = 10,
    TRACE_DELTASTEP = 11,
    TRACE_NEARFAR = 12
};

static inline bool is_trace_mode(ColorMode mode)
{
    return (static_cast<uint32_t>(mode) & static_cast<uint32_t>(ColorMode::TRACE)) != 0;
}

struct PushConstants
{
    float offset_x;
    float offset_y;
    float scale_x;
    float scale_y;
    float point_size;
};

struct App
{
    // UI state
    double pan_x = 0.0;
    double pan_y = 0.0;
    double zoom = 1.0;
    double point_size = 3.0;
    double last_mouse_x = 0.0;
    double last_mouse_y = 0.0;
    bool is_dragging = false;
    bool white_background = false;
    std::unique_ptr<FrameRecorder> recorder;

    // Vulkan context pointer (for framebuffer resize callback)
    gpu::VulkanGraphicsContext *context = nullptr;

    // Shared state (protected by color_mutex)
    std::mutex color_mutex;
    std::condition_variable color_cv;
    bool color_update_requested = false;
    ColorMode color_mode = ColorMode::FIXED;
    uint32_t grouping_size = 64;
    uint32_t delta = 3600;

    // SSSP thread communication
    std::mutex sssp_mutex;
    std::condition_variable sssp_cv;
    bool sssp_run_requested = false;

    // Tracers for each algorithm (only one is actively used at runtime)
    DeltaStepTracer deltastep_tracer;
    NearFarTracer nearfar_tracer;

    AlgorithmType algorithm_type = AlgorithmType::DELTA_STEP;

    // Source and target nodes (nullopt = random)
    std::optional<uint32_t> src_node;
    std::optional<uint32_t> dst_node;
};

class DeltaStepRunner : public AlgorithmRunner
{
  public:
    DeltaStepRunner(const gpu::GraphBuffers<common::WeightedGraph<uint32_t>> &graph_buffers,
                    std::unique_ptr<gpu::DeltaStepBuffers> buffers_ptr_, // NOLINT
                    vk::Device device,
                    gpu::Statistics &statistics,
                    uint32_t delta,
                    uint32_t batch_size)
        : graph_buffers(graph_buffers), buffers_ptr(std::move(buffers_ptr_)), device(device),
          statistics(statistics), delta_value(delta),
          algorithm(graph_buffers, *buffers_ptr, device, statistics, delta, batch_size)
    {
    }

    void initialize(vk::CommandPool cmd_pool) override { algorithm.initialize(cmd_pool); }

    uint32_t run_with_trace(vk::CommandPool cmd_pool,
                            gpu::SharedQueue &queue,
                            uint32_t src_node,
                            uint32_t dst_node,
                            App &app) override
    {
        return algorithm.run(cmd_pool, queue, src_node, dst_node, &app.deltastep_tracer);
    }

    std::optional<VisualizationPayload> payload(App &app) const override
    {
        auto payload = app.deltastep_tracer.payload();
        if (payload)
        {
            return VisualizationPayload{*payload};
        }
        return std::nullopt;
    }

    AlgorithmType type() const override { return AlgorithmType::DELTA_STEP; }
    vk::Buffer dist_buffer() const override { return buffers_ptr->dist_buffer(); }
    std::array<vk::Buffer, 2> changed_buffers() const override
    {
        return buffers_ptr->changed_buffers();
    }
    std::array<vk::Buffer, 2> near_buffers() const override
    {
        return {VK_NULL_HANDLE, VK_NULL_HANDLE};
    }
    std::array<vk::Buffer, 2> far_buffers() const override
    {
        return {VK_NULL_HANDLE, VK_NULL_HANDLE};
    }
    uint32_t max_distance() const override { return *buffers_ptr->max_distance(); }
    uint32_t delta() const override { return delta_value; }

  private:
    const gpu::GraphBuffers<common::WeightedGraph<uint32_t>> &graph_buffers;
    std::unique_ptr<gpu::DeltaStepBuffers> buffers_ptr;
    vk::Device device;
    gpu::Statistics &statistics;
    uint32_t delta_value;
    gpu::DeltaStep<common::WeightedGraph<uint32_t>> algorithm;
};

class NearFarRunner : public AlgorithmRunner
{
  public:
    NearFarRunner(const gpu::GraphBuffers<common::WeightedGraph<uint32_t>> &graph_buffers,
                  std::unique_ptr<gpu::NearFarBuffers> buffers_ptr_, // NOLINT
                  vk::Device device,
                  gpu::Statistics &statistics,
                  uint32_t delta,
                  uint32_t batch_size)
        : graph_buffers(graph_buffers), buffers_ptr(std::move(buffers_ptr_)), device(device),
          statistics(statistics), delta_value(delta),
          algorithm(graph_buffers, *buffers_ptr, device, statistics, delta, batch_size)
    {
    }

    void initialize(vk::CommandPool cmd_pool) override { algorithm.initialize(cmd_pool); }

    uint32_t run_with_trace(vk::CommandPool cmd_pool,
                            gpu::SharedQueue &queue,
                            uint32_t src_node,
                            uint32_t dst_node,
                            App &app) override
    {
        return algorithm.run(cmd_pool, queue, src_node, dst_node, &app.nearfar_tracer);
    }

    std::optional<VisualizationPayload> payload(App &app) const override
    {
        auto payload = app.nearfar_tracer.payload();
        if (payload)
        {
            phase = payload->phase_index;
            return VisualizationPayload{*payload};
        }
        return std::nullopt;
    }

    AlgorithmType type() const override { return AlgorithmType::NEAR_FAR; }
    vk::Buffer dist_buffer() const override { return buffers_ptr->dist_buffer(); }
    std::array<vk::Buffer, 2> changed_buffers() const override
    {
        return {VK_NULL_HANDLE, VK_NULL_HANDLE};
    }
    std::array<vk::Buffer, 2> near_buffers() const override { return buffers_ptr->near_buffers(); }
    std::array<vk::Buffer, 2> far_buffers() const override { return buffers_ptr->far_buffers(); }
    uint32_t max_distance() const override { return (phase + 1) * delta_value; }
    uint32_t delta() const override { return delta_value; }

  private:
    const gpu::GraphBuffers<common::WeightedGraph<uint32_t>> &graph_buffers;
    std::unique_ptr<gpu::NearFarBuffers> buffers_ptr;
    vk::Device device;
    gpu::Statistics &statistics;
    uint32_t delta_value;
    mutable uint32_t phase = 0;
    gpu::NearFar<common::WeightedGraph<uint32_t>> algorithm;
};

static void notify_color_update(App &app)
{
    std::scoped_lock lock(app.color_mutex);
    app.color_update_requested = true;
    app.color_cv.notify_one();
}

static void scroll_callback(GLFWwindow *window, double, double yoffset)
{
    auto *app = static_cast<App *>(glfwGetWindowUserPointer(window));
    double zoom_factor = 1.1;
    if (yoffset > 0)
    {
        app->zoom *= zoom_factor;
    }
    else if (yoffset < 0)
    {
        app->zoom /= zoom_factor;
    }
}

static void mouse_button_callback(GLFWwindow *window, int button, int action, int)
{
    auto *app = static_cast<App *>(glfwGetWindowUserPointer(window));
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS)
        {
            app->is_dragging = true;
            glfwGetCursorPos(window, &app->last_mouse_x, &app->last_mouse_y);
        }
        else if (action == GLFW_RELEASE)
        {
            app->is_dragging = false;
        }
    }
}

static void cursor_pos_callback(GLFWwindow *window, double xpos, double ypos)
{
    auto *app = static_cast<App *>(glfwGetWindowUserPointer(window));
    if (app->is_dragging)
    {
        double dx = xpos - app->last_mouse_x;
        double dy = ypos - app->last_mouse_y;

        app->pan_x += dx / app->zoom;
        app->pan_y -= dy / app->zoom;

        app->last_mouse_x = xpos;
        app->last_mouse_y = ypos;
    }
}

static void key_callback(GLFWwindow *window, int key, int, int action, int)
{
    auto *app = static_cast<App *>(glfwGetWindowUserPointer(window));
    if (action == GLFW_PRESS || action == GLFW_REPEAT)
    {
        if (key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD)
        {
            app->point_size = std::min(app->point_size * 1.2, 100.0);
        }
        else if (key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT)
        {
            app->point_size = std::max(app->point_size / 1.2, 0.1);
        }
    }
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_M)
        {
            ColorMode new_mode;
            uint32_t new_grouping;
            {
                std::scoped_lock lock(app->color_mutex);
                new_mode = app->color_mode;
                new_grouping = app->grouping_size;

                if (new_mode == ColorMode::FIXED)
                {
                    common::log() << "Coloring by node order" << '\n';
                    new_mode = ColorMode::ORDERING;
                }
                else if (new_mode == ColorMode::ORDERING)
                {
                    new_mode = ColorMode::WORKGROUP;
                    new_grouping = 32;
                    common::log() << "Coloring by workgroup (32 nodes)" << '\n';
                }
                else if (new_mode == ColorMode::WORKGROUP)
                {
                    if (new_grouping == 32)
                    {
                        new_grouping = 64;
                    }
                    else if (new_grouping == 64)
                    {
                        new_grouping = 128;
                    }
                    else if (new_grouping == 128)
                    {
                        new_grouping = 256;
                    }
                    else
                    {
                        new_mode = ColorMode::FIXED;
                        common::log() << "Coloring: Fixed" << '\n';
                    }

                    if (new_mode == ColorMode::WORKGROUP)
                    {
                        common::log()
                            << "Coloring by workgroup (" << new_grouping << " nodes)" << '\n';
                    }
                }
                else
                {
                    new_mode = ColorMode::FIXED;
                }

                app->color_mode = new_mode;
                app->grouping_size = new_grouping;
                app->color_update_requested = true;
                app->color_cv.notify_one();
            }
        }
        else if (key == GLFW_KEY_B)
        {
            app->white_background = !app->white_background;
        }
        else if (key == GLFW_KEY_T)
        {
            ColorMode new_mode;
            bool entering_trace = false;
            {
                std::scoped_lock lock(app->color_mutex);
                ColorMode current = app->color_mode;

                if (app->algorithm_type == AlgorithmType::DELTA_STEP)
                {
                    if (current == ColorMode::TRACE_DISTANCE)
                    {
                        common::log() << "Tracing buckets" << '\n';
                        new_mode = ColorMode::TRACE_BUCKET;
                    }
                    else if (current == ColorMode::TRACE_BUCKET)
                    {
                        common::log() << "Tracing changed nodes" << '\n';
                        new_mode = ColorMode::TRACE_DELTASTEP;
                    }
                    else
                    {
                        common::log() << "Tracing distance" << '\n';
                        new_mode = ColorMode::TRACE_DISTANCE;
                        entering_trace = !is_trace_mode(current);
                    }
                }
                else
                {
                    if (current == ColorMode::TRACE_DISTANCE)
                    {
                        common::log() << "Tracing buckets (phases)" << '\n';
                        new_mode = ColorMode::TRACE_BUCKET;
                    }
                    else if (current == ColorMode::TRACE_BUCKET)
                    {
                        common::log() << "Tracing near/far buckets" << '\n';
                        new_mode = ColorMode::TRACE_NEARFAR;
                    }
                    else
                    {
                        common::log() << "Tracing distance" << '\n';
                        new_mode = ColorMode::TRACE_DISTANCE;
                        entering_trace = !is_trace_mode(current);
                    }
                }

                app->color_mode = new_mode;
                app->color_update_requested = true;
                app->color_cv.notify_one();
            }

            if (entering_trace)
            {
                common::log() << "Entering Trace mode" << '\n';
                std::scoped_lock lock(app->sssp_mutex);
                app->sssp_run_requested = true;
                app->sssp_cv.notify_one();
            }
        }
        else if (key == GLFW_KEY_S)
        {
            if (is_trace_mode(app->color_mode))
            {
                if (app->algorithm_type == AlgorithmType::DELTA_STEP)
                {
                    app->deltastep_tracer.step();
                }
                else
                {
                    app->nearfar_tracer.step();
                }
            }
        }
        else if (key == GLFW_KEY_A)
        {
            if (is_trace_mode(app->color_mode))
            {
                if (app->algorithm_type == AlgorithmType::DELTA_STEP)
                {
                    if (app->deltastep_tracer.is_auto_playing())
                    {
                        app->deltastep_tracer.stop_auto_play();
                    }
                    else
                    {
                        app->deltastep_tracer.start_auto_play(500);
                    }
                }
                else
                {
                    if (app->nearfar_tracer.is_auto_playing())
                    {
                        app->nearfar_tracer.stop_auto_play();
                    }
                    else
                    {
                        app->nearfar_tracer.start_auto_play(500);
                    }
                }
            }
        }
        else if (key == GLFW_KEY_LEFT_BRACKET)
        {
            if (is_trace_mode(app->color_mode))
            {
                if (app->algorithm_type == AlgorithmType::DELTA_STEP)
                {
                    uint32_t new_interval = std::min(
                        app->deltastep_tracer.auto_play_interval() * 3 / 2, uint32_t{5000});
                    app->deltastep_tracer.set_auto_play_speed(new_interval);
                }
                else
                {
                    uint32_t new_interval =
                        std::min(app->nearfar_tracer.auto_play_interval() * 3 / 2, uint32_t{5000});
                    app->nearfar_tracer.set_auto_play_speed(new_interval);
                }
            }
        }
        else if (key == GLFW_KEY_RIGHT_BRACKET)
        {
            if (is_trace_mode(app->color_mode))
            {
                if (app->algorithm_type == AlgorithmType::DELTA_STEP)
                {
                    uint32_t new_interval =
                        std::max(app->deltastep_tracer.auto_play_interval() * 2 / 3, uint32_t{50});
                    app->deltastep_tracer.set_auto_play_speed(new_interval);
                }
                else
                {
                    uint32_t new_interval =
                        std::max(app->nearfar_tracer.auto_play_interval() * 2 / 3, uint32_t{50});
                    app->nearfar_tracer.set_auto_play_speed(new_interval);
                }
            }
        }
        else if (key == GLFW_KEY_C)
        {
            if (is_trace_mode(app->color_mode))
            {
                if (app->algorithm_type == AlgorithmType::DELTA_STEP)
                {
                    app->deltastep_tracer.continue_to_end();
                }
                else
                {
                    app->nearfar_tracer.continue_to_end();
                }
            }
        }
        else if (key == GLFW_KEY_R)
        {
            if (is_trace_mode(app->color_mode))
            {
                if (app->algorithm_type == AlgorithmType::DELTA_STEP)
                {
                    app->deltastep_tracer.continue_to_end();
                }
                else
                {
                    app->nearfar_tracer.continue_to_end();
                }

                {
                    std::scoped_lock lock(app->sssp_mutex);
                    app->sssp_run_requested = true;
                    app->sssp_cv.notify_one();
                }
            }
        }
        else if (key == GLFW_KEY_F)
        {
            if (app->recorder)
            {
                if (app->recorder->is_recording())
                {
                    app->recorder->stop_recording();
                }
                else
                {
                    app->recorder->start_recording(app->context->swapchain_extent());
                }
            }
            else
            {
                common::log_warning()
                    << "Can't start recording. No output directory specified." << '\n';
            }
        }
    }
}

static void framebuffer_resize_callback(GLFWwindow *window, int, int)
{
    auto *app = static_cast<App *>(glfwGetWindowUserPointer(window));
    if (app && app->context)
    {
        app->context->set_framebuffer_resized();
    }
}

static void setup_glfw_callbacks(GLFWwindow *window, App &app)
{
    glfwSetWindowUserPointer(window, &app);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_resize_callback);
}

static std::jthread start_sssp_thread(App &ctx,
                                      gpu::VulkanGraphicsContext &context,
                                      AlgorithmRunner &algorithm,
                                      size_t num_nodes,
                                      std::latch &initialization_latch)
{
    return std::jthread(
        [&ctx, &algorithm, &context, num_nodes, &initialization_latch](
            const std::stop_token &st) mutable
        {
            vk::CommandPoolCreateInfo pool_info(vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                                0);
            vk::CommandPool cmd_pool = context.device().createCommandPool(pool_info);

            algorithm.initialize(cmd_pool);

            initialization_latch.count_down();

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<uint32_t> dist(0, static_cast<uint32_t>(num_nodes - 1));

            while (!st.stop_requested())
            {
                {
                    std::unique_lock<std::mutex> lock(ctx.sssp_mutex);
                    ctx.sssp_cv.wait(lock,
                                     [&ctx, &st]
                                     { return ctx.sssp_run_requested || st.stop_requested(); });

                    if (st.stop_requested())
                    {
                        break;
                    }

                    ctx.sssp_run_requested = false;
                }

                uint32_t src_node = ctx.src_node.value_or(dist(gen));
                uint32_t dst_node = ctx.dst_node.value_or(dist(gen));

                common::log() << "SSSP thread: Running "
                              << (algorithm.type() == AlgorithmType::DELTA_STEP ? "delta-stepping"
                                                                                : "near-far")
                              << " from node " << src_node << '\n';

                uint32_t result = algorithm.run_with_trace(
                    cmd_pool, context.shared_queue(), src_node, dst_node, ctx);

                {
                    std::scoped_lock lock(ctx.color_mutex);
                    ctx.color_update_requested = true;
                    ctx.color_cv.notify_one();
                }

                common::log() << "SSSP thread: Completed with result " << result << '\n';
            }

            context.device().destroyCommandPool(cmd_pool);
            common::log() << "SSSP thread: Shutting down" << '\n';
        });
}

static std::jthread start_color_updater_thread(App &ctx,
                                               gpu::VulkanGraphicsContext &context,
                                               AlgorithmRunner &algorithm,
                                               vk::Buffer color_buffer,
                                               size_t num_nodes,
                                               std::latch &initialization_latch)
{
    return std::jthread(
        [&ctx, &algorithm, &context, color_buffer, num_nodes, &initialization_latch](
            const std::stop_token &st) mutable
        {
            vk::CommandPoolCreateInfo pool_info(vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                                0);
            vk::CommandPool cmd_pool = context.device().createCommandPool(pool_info);

            struct NodeColorPushConsts
            {
                uint32_t num_nodes;
                uint32_t max_distance;
                uint32_t color_mode;
                uint32_t delta;
                uint32_t bucket_index;
                uint32_t changed_buffer_index;
                uint32_t grouping_size;
                uint32_t near_buffer_index;
                uint32_t far_buffer_index;
            };

            auto [changed_0, changed_1] = algorithm.changed_buffers();
            auto [near_0, near_1] = algorithm.near_buffers();
            auto [far_0, far_1] = algorithm.far_buffers();

            std::vector<vk::Buffer> descriptor_buffers = {algorithm.dist_buffer(),
                                                          color_buffer,
                                                          changed_0,
                                                          changed_1,
                                                          near_0,
                                                          near_1,
                                                          far_0,
                                                          far_1};

            auto node_color_pipeline = gpu::create_compute_pipeline<NodeColorPushConsts>(
                context.device(), "node_color.spv", {descriptor_buffers});

            initialization_latch.count_down();

            auto dispatch_shader = [&](const ColorMode mode_enum,
                                       const uint32_t grouping_size,
                                       const std::optional<VisualizationPayload> &maybe_payload)
            {
                NodeColorPushConsts pc{.num_nodes = static_cast<uint32_t>(num_nodes),
                                       .max_distance = algorithm.max_distance(),
                                       .color_mode = static_cast<uint32_t>(mode_enum),
                                       .delta = algorithm.delta(),
                                       .bucket_index = UINT32_MAX,
                                       .changed_buffer_index = UINT32_MAX,
                                       .grouping_size = grouping_size,
                                       .near_buffer_index = UINT32_MAX,
                                       .far_buffer_index = UINT32_MAX};

                if (maybe_payload)
                {
                    if (std::holds_alternative<DeltaStepPayload>(*maybe_payload))
                    {
                        const auto &payload = std::get<DeltaStepPayload>(*maybe_payload);
                        pc.bucket_index = payload.bucket_index;
                        pc.changed_buffer_index = payload.buffer_index;
                    }
                    else
                    {
                        const auto &payload = std::get<NearFarPayload>(*maybe_payload);
                        pc.bucket_index = payload.phase_index;
                        pc.near_buffer_index = payload.near_buffer_index;
                        pc.far_buffer_index = payload.far_buffer_index;
                    }
                }

                auto device = context.device();
                vk::CommandBuffer cmd = device.allocateCommandBuffers(
                    {cmd_pool, vk::CommandBufferLevel::ePrimary, 1})[0];

                cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
                cmd.bindPipeline(vk::PipelineBindPoint::eCompute, node_color_pipeline.pipeline);
                cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                       node_color_pipeline.layout,
                                       0,
                                       1,
                                       node_color_pipeline.descriptor_sets.data(),
                                       0,
                                       nullptr);
                cmd.pushConstants(node_color_pipeline.layout,
                                  vk::ShaderStageFlagBits::eCompute,
                                  0,
                                  sizeof(pc),
                                  &pc);

                uint32_t num_workgroups = (static_cast<uint32_t>(num_nodes) + 255) / 256;
                cmd.dispatch(num_workgroups, 1, 1);

                vk::BufferMemoryBarrier barrier(vk::AccessFlagBits::eShaderWrite,
                                                vk::AccessFlagBits::eVertexAttributeRead,
                                                VK_QUEUE_FAMILY_IGNORED,
                                                VK_QUEUE_FAMILY_IGNORED,
                                                color_buffer,
                                                0,
                                                VK_WHOLE_SIZE);

                cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                    vk::PipelineStageFlagBits::eVertexInput,
                                    {},
                                    {},
                                    barrier,
                                    {});

                cmd.end();

                vk::Fence fence = device.createFence({});
                context.shared_queue().submit({{0, nullptr, nullptr, 1, &cmd}}, fence);
                (void)device.waitForFences(1, &fence, VK_TRUE, UINT64_MAX);

                device.destroyFence(fence);
                device.freeCommandBuffers(cmd_pool, 1, &cmd);
            };

            while (!st.stop_requested())
            {
                ColorMode mode;
                uint32_t grouping_size;
                {
                    std::unique_lock<std::mutex> lock(ctx.color_mutex);
                    ctx.color_cv.wait(
                        lock,
                        [&ctx, &st] { return ctx.color_update_requested || st.stop_requested(); });

                    if (st.stop_requested())
                    {
                        break;
                    }

                    ctx.color_update_requested = false;
                    mode = ctx.color_mode;
                    grouping_size = ctx.grouping_size;
                }

                if (is_trace_mode(mode))
                {
                    while (!st.stop_requested())
                    {
                        {
                            std::scoped_lock lock(ctx.color_mutex);
                            mode = ctx.color_mode;
                            grouping_size = ctx.grouping_size;
                        }

                        if (!is_trace_mode(mode))
                        {
                            break;
                        }

                        bool is_finished = (ctx.algorithm_type == AlgorithmType::DELTA_STEP)
                                               ? ctx.deltastep_tracer.is_finished()
                                               : ctx.nearfar_tracer.is_finished();

                        if (is_finished)
                        {
                            auto payload = algorithm.payload(ctx);
                            dispatch_shader(mode, grouping_size, payload);
                            break;
                        }

                        bool has_signal = (ctx.algorithm_type == AlgorithmType::DELTA_STEP)
                                              ? ctx.deltastep_tracer.wait_for_signal(100)
                                              : ctx.nearfar_tracer.wait_for_signal(100);

                        if (has_signal)
                        {
                            auto payload = algorithm.payload(ctx);
                            dispatch_shader(mode, grouping_size, payload);
                        }
                    }
                }
                else
                {
                    dispatch_shader(mode, grouping_size, {});
                }
            }

            node_color_pipeline.destroy(context.device());
            context.device().destroyCommandPool(cmd_pool);
            common::log() << "Color updater thread: Shutting down" << '\n';
        });
}

static std::pair<vk::Buffer, vk::DeviceMemory> create_color_buffer(
    vk::Device device, vk::PhysicalDeviceMemoryProperties mem_props, size_t num_nodes)
{
    vk::Buffer color_buffer = gpu::create_exclusive_buffer<float>(
        device,
        num_nodes * 3,
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer);

    vk::DeviceMemory color_buffer_memory = gpu::alloc_and_bind(
        device, mem_props, color_buffer, vk::MemoryPropertyFlagBits::eDeviceLocal);

    return {color_buffer, color_buffer_memory};
}

static void project_coordinates(gpu::CoordinatesBuffer &coord_buffer,
                                vk::Device device,
                                vk::CommandPool cmd_pool,
                                gpu::SharedQueue &queue,
                                const common::WebMercatorPoint &min_point,
                                const common::WebMercatorPoint &max_point,
                                size_t num_coords)
{
    struct ProjectionPushConstants
    {
        float center_x;
        float center_y;
        float inv_width;
        float inv_height;
        uint32_t num_points;
    };

    float center_x = (min_point.x + max_point.x) / 2.0f;
    float center_y = (min_point.y + max_point.y) / 2.0f;
    float inv_width = 1.0f / (max_point.x - min_point.x);
    float inv_height = 1.0f / (max_point.y - min_point.y);

    auto coord_buffers = coord_buffer.buffers();

    auto compute_pipeline = gpu::create_compute_pipeline<ProjectionPushConstants>(
        device, "project_coordinates.spv", {{coord_buffers[0], coord_buffers[1]}});

    vk::CommandBuffer compute_cmd =
        device.allocateCommandBuffers({cmd_pool, vk::CommandBufferLevel::ePrimary, 1})[0];

    compute_cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    compute_cmd.bindPipeline(vk::PipelineBindPoint::eCompute, compute_pipeline.pipeline);
    compute_cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                   compute_pipeline.layout,
                                   0,
                                   1,
                                   compute_pipeline.descriptor_sets.data(),
                                   0,
                                   nullptr);

    ProjectionPushConstants push_constants{.center_x = center_x,
                                           .center_y = center_y,
                                           .inv_width = inv_width,
                                           .inv_height = inv_height,
                                           .num_points = static_cast<uint32_t>(num_coords)};

    compute_cmd.pushConstants(compute_pipeline.layout,
                              vk::ShaderStageFlagBits::eCompute,
                              0,
                              sizeof(ProjectionPushConstants),
                              &push_constants);

    uint32_t num_workgroups = (static_cast<uint32_t>(num_coords) + 255) / 256;
    compute_cmd.dispatch(num_workgroups, 1, 1);

    vk::BufferMemoryBarrier barrier(vk::AccessFlagBits::eShaderWrite,
                                    vk::AccessFlagBits::eVertexAttributeRead,
                                    VK_QUEUE_FAMILY_IGNORED,
                                    VK_QUEUE_FAMILY_IGNORED,
                                    coord_buffers[1],
                                    0,
                                    VK_WHOLE_SIZE);

    compute_cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eVertexInput,
                                {},
                                {},
                                barrier,
                                {});

    compute_cmd.end();

    vk::Fence compute_fence = device.createFence({});
    queue.submit({{0, nullptr, nullptr, 1, &compute_cmd}}, compute_fence);
    (void)device.waitForFences(1, &compute_fence, VK_TRUE, UINT64_MAX);

    device.destroyFence(compute_fence);
    device.freeCommandBuffers(cmd_pool, 1, &compute_cmd);
    compute_pipeline.destroy(device);
}

static std::pair<vk::Pipeline, vk::PipelineLayout> create_graphics_pipeline(
    vk::Device device, vk::RenderPass render_pass, vk::Extent2D swapchain_extent)
{
    vk::ShaderModule vert_shader = gpu::create_shader_module(device, "visualize_node.vert.spv");
    vk::ShaderModule frag_shader = gpu::create_shader_module(device, "visualize_node.frag.spv");

    vk::PipelineShaderStageCreateInfo vert_stage_info(
        {}, vk::ShaderStageFlagBits::eVertex, vert_shader, "main");
    vk::PipelineShaderStageCreateInfo frag_stage_info(
        {}, vk::ShaderStageFlagBits::eFragment, frag_shader, "main");

    std::array<vk::PipelineShaderStageCreateInfo, 2> shader_stages = {vert_stage_info,
                                                                      frag_stage_info};

    std::array<vk::VertexInputBindingDescription, 2> bindings = {
        vk::VertexInputBindingDescription(0, sizeof(float) * 2, vk::VertexInputRate::eVertex),
        vk::VertexInputBindingDescription(1, sizeof(float) * 3, vk::VertexInputRate::eVertex)};

    std::array<vk::VertexInputAttributeDescription, 2> attributes = {
        vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, 0),
        vk::VertexInputAttributeDescription(1, 1, vk::Format::eR32G32B32Sfloat, 0)};

    vk::PipelineVertexInputStateCreateInfo vertex_input_info(
        {},
        static_cast<uint32_t>(bindings.size()),
        bindings.data(),
        static_cast<uint32_t>(attributes.size()),
        attributes.data());

    vk::PipelineInputAssemblyStateCreateInfo input_assembly(
        {}, vk::PrimitiveTopology::ePointList, VK_FALSE);

    vk::Viewport viewport(0.0f,
                          0.0f,
                          static_cast<float>(swapchain_extent.width),
                          static_cast<float>(swapchain_extent.height),
                          0.0f,
                          1.0f);
    vk::Rect2D scissor({0, 0}, swapchain_extent);

    vk::PipelineViewportStateCreateInfo viewport_state({}, 1, &viewport, 1, &scissor);

    vk::PipelineRasterizationStateCreateInfo rasterizer({},
                                                        VK_FALSE,
                                                        VK_FALSE,
                                                        vk::PolygonMode::eFill,
                                                        vk::CullModeFlagBits::eNone,
                                                        vk::FrontFace::eClockwise,
                                                        VK_FALSE,
                                                        0.0f,
                                                        0.0f,
                                                        0.0f,
                                                        1.0f);

    vk::PipelineMultisampleStateCreateInfo multisampling(
        {}, vk::SampleCountFlagBits::e1, VK_FALSE, 1.0f, nullptr, VK_FALSE, VK_FALSE);

    vk::PipelineColorBlendAttachmentState color_blend_attachment(
        VK_TRUE,
        vk::BlendFactor::eSrcAlpha,
        vk::BlendFactor::eOneMinusSrcAlpha,
        vk::BlendOp::eAdd,
        vk::BlendFactor::eOne,
        vk::BlendFactor::eZero,
        vk::BlendOp::eAdd,
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);

    vk::PipelineColorBlendStateCreateInfo color_blending(
        {}, VK_FALSE, vk::LogicOp::eCopy, 1, &color_blend_attachment);

    vk::PushConstantRange push_constant_range(
        vk::ShaderStageFlagBits::eVertex, 0, sizeof(PushConstants));

    vk::PipelineLayoutCreateInfo pipeline_layout_info({}, 0, nullptr, 1, &push_constant_range);

    vk::PipelineLayout pipeline_layout = device.createPipelineLayout(pipeline_layout_info);

    std::vector<vk::DynamicState> dynamic_states = {vk::DynamicState::eViewport,
                                                    vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynamic_state(
        {}, static_cast<uint32_t>(dynamic_states.size()), dynamic_states.data());

    vk::GraphicsPipelineCreateInfo pipeline_info({},
                                                 static_cast<uint32_t>(shader_stages.size()),
                                                 shader_stages.data(),
                                                 &vertex_input_info,
                                                 &input_assembly,
                                                 nullptr,
                                                 &viewport_state,
                                                 &rasterizer,
                                                 &multisampling,
                                                 nullptr,
                                                 &color_blending,
                                                 &dynamic_state,
                                                 pipeline_layout,
                                                 render_pass,
                                                 0);

    auto pipeline_result = device.createGraphicsPipeline(nullptr, pipeline_info);
    if (pipeline_result.result != vk::Result::eSuccess)
    {
        throw std::runtime_error("Failed to create graphics pipeline");
    }
    vk::Pipeline graphics_pipeline = pipeline_result.value;

    device.destroyShaderModule(vert_shader);
    device.destroyShaderModule(frag_shader);

    return {graphics_pipeline, pipeline_layout};
}

static PushConstants calculate_view_transform(const App &camera,
                                              vk::Extent2D swapchain_extent,
                                              double graph_width,
                                              double graph_height)
{
    double aspect_ratio = static_cast<double>(swapchain_extent.width) / swapchain_extent.height;
    double graph_aspect = graph_width / graph_height;

    double scale_x;
    double scale_y;
    if (graph_aspect > aspect_ratio)
    {
        scale_x = camera.zoom;
        scale_y = camera.zoom * aspect_ratio / graph_aspect;
    }
    else
    {
        scale_x = camera.zoom * graph_aspect / aspect_ratio;
        scale_y = camera.zoom;
    }

    double normalized_pan_x = camera.pan_x / swapchain_extent.width * 2.0;
    double normalized_pan_y = camera.pan_y / swapchain_extent.height * 2.0;

    return PushConstants{.offset_x = static_cast<float>(normalized_pan_x),
                         .offset_y = static_cast<float>(normalized_pan_y),
                         .scale_x = static_cast<float>(scale_x),
                         .scale_y = static_cast<float>(scale_y),
                         .point_size =
                             std::max(0.5f, static_cast<float>(camera.point_size * camera.zoom))};
}

static void record_render_commands(vk::CommandBuffer command_buffer,
                                   vk::RenderPass render_pass,
                                   vk::Framebuffer framebuffer,
                                   vk::Extent2D swapchain_extent,
                                   vk::Pipeline graphics_pipeline,
                                   vk::PipelineLayout pipeline_layout,
                                   const PushConstants &push_constants,
                                   vk::Buffer projected_coords_buffer,
                                   vk::Buffer color_buffer,
                                   uint32_t vertex_count,
                                   bool white_background)
{
    command_buffer.reset();

    vk::CommandBufferBeginInfo begin_info;
    command_buffer.begin(begin_info);

    float bg_color = white_background ? 1.0f : 0.0f;
    vk::ClearValue clear_color(
        vk::ClearColorValue(std::array<float, 4>{bg_color, bg_color, bg_color, 0.0f}));

    vk::RenderPassBeginInfo render_pass_info(
        render_pass, framebuffer, {{0, 0}, swapchain_extent}, 1, &clear_color);

    command_buffer.beginRenderPass(render_pass_info, vk::SubpassContents::eInline);

    command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphics_pipeline);

    vk::Viewport viewport(0.0f,
                          0.0f,
                          static_cast<float>(swapchain_extent.width),
                          static_cast<float>(swapchain_extent.height),
                          0.0f,
                          1.0f);
    command_buffer.setViewport(0, 1, &viewport);

    vk::Rect2D scissor({0, 0}, swapchain_extent);
    command_buffer.setScissor(0, 1, &scissor);

    command_buffer.pushConstants(pipeline_layout,
                                 vk::ShaderStageFlagBits::eVertex,
                                 0,
                                 sizeof(PushConstants),
                                 &push_constants);

    std::array<vk::Buffer, 2> vertex_buffers = {projected_coords_buffer, color_buffer};
    std::array<vk::DeviceSize, 2> offsets = {0, 0};
    command_buffer.bindVertexBuffers(0, 2, vertex_buffers.data(), offsets.data());

    command_buffer.draw(vertex_count, 1, 0, 0);

    command_buffer.endRenderPass();
    command_buffer.end();
}

struct RenderPipeline
{
    vk::Buffer color_buffer;
    vk::DeviceMemory color_buffer_memory;
    vk::Pipeline graphics_pipeline;
    vk::PipelineLayout pipeline_layout;
    std::vector<vk::CommandBuffer> command_buffers;

    void destroy(vk::Device device)
    {
        device.destroyPipeline(graphics_pipeline);
        device.destroyPipelineLayout(pipeline_layout);
        device.destroyBuffer(color_buffer);
        device.freeMemory(color_buffer_memory);
    }
};

static auto load_graph(const std::string &base_path)
{
    common::log() << "Loading graph from: " << base_path << '\n';

    auto graph = common::files::read_weighted_graph<uint32_t>(base_path);
    auto coordinates = common::files::read_coordinates(base_path);

    if (graph.num_nodes() != coordinates.size())
    {
        throw std::runtime_error("Graph node count (" + std::to_string(graph.num_nodes()) +
                                 ") does not match coordinate count (" +
                                 std::to_string(coordinates.size()) + ")");
    }

    common::log() << "Loaded graph with " << graph.num_nodes() << " nodes and " << graph.num_edges()
                  << " edges" << '\n';

    auto bounding_box = common::bounds(coordinates);

    return std::make_tuple(std::move(graph), std::move(coordinates), bounding_box);
}

static RenderPipeline setup_render_pipeline(gpu::VulkanGraphicsContext &context,
                                            gpu::CoordinatesBuffer &coord_buffer,
                                            const uint32_t num_nodes,
                                            const common::BoundingBox &bounding_box)
{
    vk::Device device = context.device();
    vk::PhysicalDeviceMemoryProperties mem_props = context.physical_device().getMemoryProperties();
    vk::CommandPool cmd_pool = context.command_pool();
    auto &queue = context.shared_queue();

    auto min_point = common::to_web_mercator(bounding_box.south_east);
    auto max_point = common::to_web_mercator(bounding_box.north_west);

    auto [color_buffer, color_buffer_memory] = create_color_buffer(device, mem_props, num_nodes);

    project_coordinates(coord_buffer, device, cmd_pool, queue, min_point, max_point, num_nodes);

    auto [graphics_pipeline, pipeline_layout] =
        create_graphics_pipeline(device, context.render_pass(), context.swapchain_extent());

    constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;
    std::vector<vk::CommandBuffer> command_buffers = device.allocateCommandBuffers(
        {cmd_pool, vk::CommandBufferLevel::ePrimary, MAX_FRAMES_IN_FLIGHT});

    return {color_buffer,
            color_buffer_memory,
            graphics_pipeline,
            pipeline_layout,
            std::move(command_buffers)};
}

static std::pair<std::jthread, std::jthread>
start_workers(App &app,
              gpu::VulkanGraphicsContext &context,
              RenderPipeline &render_pipeline,
              AlgorithmRunner &algorithm,
              size_t num_nodes,
              const std::optional<std::string> &output_dir)
{
    common::log() << "Starting worker threads..." << '\n';
    std::latch initialization_latch(2);
    auto sssp_thread = start_sssp_thread(app, context, algorithm, num_nodes, initialization_latch);
    auto color_thread = start_color_updater_thread(
        app, context, algorithm, render_pipeline.color_buffer, num_nodes, initialization_latch);

    initialization_latch.wait();

    notify_color_update(app);

    if (output_dir)
    {
        common::log() << "Frame recording enabled. Press 'F' to start/stop recording." << '\n';
        app.recorder = std::make_unique<FrameRecorder>(context, *output_dir);
    }

    return {std::move(sssp_thread), std::move(color_thread)};
}

static void run_render_loop(App &app,
                            gpu::VulkanGraphicsContext &context,
                            RenderPipeline &graphics_pipeline,
                            gpu::CoordinatesBuffer &coord_buffer,
                            uint32_t num_nodes,
                            const common::BoundingBox bounding_box)
{
    const auto min_point = common::to_web_mercator(bounding_box.south_east);
    const auto max_point = common::to_web_mercator(bounding_box.north_west);

    common::log() << "Starting render loop..." << '\n';

    auto coord_buffers = coord_buffer.buffers();
    uint32_t current_frame = 0;

    while (!context.should_close())
    {
        context.poll_events();

        auto frame = context.begin_frame();

        PushConstants push_constants = calculate_view_transform(
            app, context.swapchain_extent(), max_point.x - min_point.x, max_point.y - min_point.y);

        vk::CommandBuffer command_buffer = graphics_pipeline.command_buffers[current_frame];

        record_render_commands(command_buffer,
                               context.render_pass(),
                               frame.framebuffer,
                               context.swapchain_extent(),
                               graphics_pipeline.graphics_pipeline,
                               graphics_pipeline.pipeline_layout,
                               push_constants,
                               coord_buffers[1],
                               graphics_pipeline.color_buffer,
                               num_nodes,
                               app.white_background);

        vk::PipelineStageFlags wait_stages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
        vk::SubmitInfo submit_info(
            1, &frame.image_available, wait_stages, 1, &command_buffer, 1, &frame.render_finished);

        context.shared_queue().submit(1, &submit_info, frame.in_flight);

        if (app.recorder && app.recorder->is_recording())
        {
            (void)context.device().waitForFences(1, &frame.in_flight, VK_TRUE, UINT64_MAX);
            app.recorder->capture_frame(frame.image, context.swapchain_extent());
        }

        context.end_frame(frame);

        current_frame = (current_frame + 1) % 2;
    }
}

static void shutdown(App &app,
                     std::jthread &sssp_thread,
                     std::jthread &color_thread,
                     gpu::VulkanGraphicsContext &context,
                     RenderPipeline &graphics_pipeline)
{
    context.device().waitIdle();

    common::log() << "Shutting down worker threads..." << '\n';

    sssp_thread.request_stop();
    color_thread.request_stop();

    app.sssp_cv.notify_all();
    app.color_cv.notify_all();

    if (app.algorithm_type == AlgorithmType::DELTA_STEP)
    {
        app.deltastep_tracer.continue_to_end();
    }
    else
    {
        app.nearfar_tracer.continue_to_end();
    }

    graphics_pipeline.destroy(context.device());
}

int main(int argc, char **argv)
{
    argparse::ArgumentParser program("visualize_graph", "1.0.0");
    program.add_description("GPU-accelerated graph visualization with SSSP algorithm tracing.");

    program.add_argument("graph_path").help("path to preprocessed graph data (without extension)");

    program.add_argument("-o", "--output").help("output directory for frame recording");

    program.add_argument("-a", "--algorithm")
        .default_value(std::string("deltastep"))
        .help("algorithm: deltastep (default) or nearfar");

    program.add_argument("-d", "--delta")
        .default_value(std::string("auto"))
        .help("delta parameter for algorithm: \"auto\" or integer");

    program.add_argument("-s", "--source")
        .default_value(std::string("random"))
        .help("source node: \"random\", node ID, or lon,lat");

    program.add_argument("-t", "--target")
        .default_value(std::string("random"))
        .help("target node: \"random\", node ID, or lon,lat");

    program.add_argument("-b", "--batch-size")
        .default_value(64u)
        .scan<'u', uint32_t>()
        .help("relaxation batch size for GPU algorithms");

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception &err)
    {
        common::log_error() << err.what() << '\n' << program;
        return EXIT_FAILURE;
    }

    std::string base_path = program.get("graph_path");
    std::optional<std::string> output_dir;
    if (program.is_used("--output"))
    {
        output_dir = program.get("--output");
    }

    std::string algo_str = program.get("--algorithm");
    AlgorithmType algorithm_type = AlgorithmType::DELTA_STEP;
    if (algo_str == "nearfar")
    {
        algorithm_type = AlgorithmType::NEAR_FAR;
    }
    else if (algo_str != "deltastep")
    {
        common::log_error() << "Unknown algorithm: " << algo_str << '\n';
        return EXIT_FAILURE;
    }

    auto delta_str = program.get("--delta");

    auto [graph, coordinates, bounding_box] = load_graph(base_path);

    auto delta = common::compute_delta_heuristic(graph);
    if (delta_str != "auto")
    {
        delta = std::stoi(delta_str);
    }
    common::log() << "Using delta value " << delta << '\n';

    auto source_str = program.get("--source");
    auto target_str = program.get("--target");
    auto batch_size = program.get<uint32_t>("--batch-size");

    common::NearestNeighbour nn(coordinates);

    auto maybe_src_coord = common::parse_coordinate(source_str);
    auto maybe_dst_coord = common::parse_coordinate(target_str);
    auto maybe_src_node_id = maybe_src_coord ? std::nullopt : common::parse_node_id(source_str);
    auto maybe_dst_node_id = maybe_dst_coord ? std::nullopt : common::parse_node_id(target_str);

    App app;
    app.algorithm_type = algorithm_type;
    app.delta = delta;

    if (maybe_src_coord)
    {
        app.src_node = nn.nearest(*maybe_src_coord);
    }
    else if (maybe_src_node_id)
    {
        app.src_node = *maybe_src_node_id;
    }

    if (maybe_dst_coord)
    {
        app.dst_node = nn.nearest(*maybe_dst_coord);
    }
    else if (maybe_dst_node_id)
    {
        app.dst_node = *maybe_dst_node_id;
    }
    gpu::VulkanGraphicsContext context("Graph Visualizer", 1280, 720, gpu::detail::select_device());
    app.context = &context;
    setup_glfw_callbacks(context.window(), app);

    gpu::GraphBuffers<common::WeightedGraph<uint32_t>> graph_buffers(graph,
                                                                     context.device(),
                                                                     context.memory_properties(),
                                                                     context.command_pool(),
                                                                     context.shared_queue());
    gpu::CoordinatesBuffer coord_buffer(coordinates,
                                        context.device(),
                                        context.memory_properties(),
                                        context.command_pool(),
                                        context.shared_queue());

    gpu::Statistics statistics(context.device(), context.memory_properties());

    std::unique_ptr<AlgorithmRunner> algorithm_runner;
    if (algorithm_type == AlgorithmType::DELTA_STEP)
    {
        algorithm_runner = std::make_unique<DeltaStepRunner>(
            graph_buffers,
            std::make_unique<gpu::DeltaStepBuffers>(
                graph.num_nodes(), context.device(), context.memory_properties()),
            context.device(),
            statistics,
            app.delta,
            batch_size);
    }
    else
    {
        algorithm_runner = std::make_unique<NearFarRunner>(
            graph_buffers,
            std::make_unique<gpu::NearFarBuffers>(
                graph.num_nodes(), context.device(), context.memory_properties()),
            context.device(),
            statistics,
            app.delta,
            batch_size);
    }

    auto render_pipeline =
        setup_render_pipeline(context, coord_buffer, graph.num_nodes(), bounding_box);

    auto [sssp_thread, color_thread] = start_workers(
        app, context, render_pipeline, *algorithm_runner, graph.num_nodes(), output_dir);

    run_render_loop(app, context, render_pipeline, coord_buffer, graph.num_nodes(), bounding_box);

    shutdown(app, sssp_thread, color_thread, context, render_pipeline);

    return EXIT_SUCCESS;
}
