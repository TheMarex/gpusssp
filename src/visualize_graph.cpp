#include "common/coordinate.hpp"
#include "common/files.hpp"
#include "common/logger.hpp"
#include "common/web_mercator.hpp"
#include "common/weighted_graph.hpp"
#include "gpu/coordinates_buffer.hpp"
#include "gpu/memory.hpp"
#include "gpu/shader.hpp"
#include "gpu/vulkan_graphics_context.hpp"

#include <GLFW/glfw3.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
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
#include "gpu/statistics.hpp"
#include "gpu/tracer.hpp"

using namespace gpusssp;

class FrameRecorder
{
  public:
    FrameRecorder(vk::Device device,
                  vk::PhysicalDeviceMemoryProperties mem_props,
                  vk::CommandPool cmd_pool,
                  gpu::SharedQueue &queue,
                  vk::Extent2D extent,
                  vk::Format format,
                  std::string output_base_dir)
        : m_device(device), m_mem_props(mem_props), m_cmd_pool(cmd_pool), m_queue(queue),
          m_extent(extent), m_format(format), m_output_base_dir(std::move(output_base_dir)),
          m_frame_number(0), m_is_recording(false), m_staging_buffer(nullptr),
          m_staging_memory(nullptr)
    {
        create_staging_buffer();
    }

    ~FrameRecorder()
    {
        if (m_is_recording)
        {
            stop_recording();
        }
        destroy_staging_buffer();
    }

    void start_recording()
    {
        if (m_is_recording)
        {
            return;
        }

        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch())
                             .count();

        m_current_recording_dir = m_output_base_dir + "/" + std::to_string(timestamp);
        std::filesystem::create_directories(m_current_recording_dir);

        m_frame_number = 0;
        m_is_recording = true;

        common::log() << "Recording started: " << m_current_recording_dir << std::endl;
    }

    void stop_recording()
    {
        if (!m_is_recording)
        {
            return;
        }

        m_is_recording = false;

        common::log() << "Recording stopped. " << m_frame_number << " frames captured."
                      << std::endl;
        common::log() << "To convert to GIF, run:" << std::endl;
        common::log() << "  ffmpeg -framerate 60 -i " << m_current_recording_dir
                      << "/frame_%06d.ppm -vf \"fps=30\" " << m_current_recording_dir
                      << "/output.gif" << std::endl;
    }

    void capture_frame(vk::Image swapchain_image)
    {
        if (!m_is_recording)
        {
            return;
        }

        copy_image_to_buffer(swapchain_image);
        save_buffer_as_ppm();
        m_frame_number++;
    }

    bool is_recording() const { return m_is_recording; }

  private:
    vk::Device m_device;
    vk::PhysicalDeviceMemoryProperties m_mem_props;
    vk::CommandPool m_cmd_pool;
    gpu::SharedQueue &m_queue;
    vk::Extent2D m_extent;
    vk::Format m_format;

    std::string m_output_base_dir;
    std::string m_current_recording_dir;
    uint32_t m_frame_number;
    bool m_is_recording;

    vk::Buffer m_staging_buffer;
    vk::DeviceMemory m_staging_memory;
    size_t m_buffer_size;

    void create_staging_buffer()
    {
        m_buffer_size = m_extent.width * m_extent.height * 4;

        m_staging_buffer = gpu::create_exclusive_buffer<uint8_t>(
            m_device, m_buffer_size, vk::BufferUsageFlagBits::eTransferDst);

        m_staging_memory = gpu::alloc_and_bind(m_device,
                                               m_mem_props,
                                               m_staging_buffer,
                                               vk::MemoryPropertyFlagBits::eHostVisible |
                                                   vk::MemoryPropertyFlagBits::eHostCoherent);
    }

    void destroy_staging_buffer()
    {
        if (m_staging_buffer)
        {
            m_device.destroyBuffer(m_staging_buffer);
            m_staging_buffer = nullptr;
        }
        if (m_staging_memory)
        {
            m_device.freeMemory(m_staging_memory);
            m_staging_memory = nullptr;
        }
    }

    void copy_image_to_buffer(vk::Image image)
    {
        vk::CommandBuffer cmd =
            m_device.allocateCommandBuffers({m_cmd_pool, vk::CommandBufferLevel::ePrimary, 1})[0];

        cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        vk::ImageMemoryBarrier barrier_to_transfer(
            vk::AccessFlagBits::eMemoryRead,
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

        vk::BufferImageCopy region(
            0,
            0,
            0,
            {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            {0, 0, 0},
            {m_extent.width, m_extent.height, 1});

        cmd.copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal, m_staging_buffer, 1, &region);

        vk::ImageMemoryBarrier barrier_to_present(
            vk::AccessFlagBits::eTransferRead,
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

        vk::Fence fence = m_device.createFence({});
        m_queue.submit({{0, nullptr, nullptr, 1, &cmd}}, fence);
        (void)m_device.waitForFences(1, &fence, VK_TRUE, UINT64_MAX);

        m_device.destroyFence(fence);
        m_device.freeCommandBuffers(m_cmd_pool, 1, &cmd);
    }

    void save_buffer_as_ppm()
    {
        std::ostringstream filename;
        filename << m_current_recording_dir << "/frame_" << std::setw(6) << std::setfill('0')
                 << m_frame_number << ".ppm";

        std::ofstream file(filename.str(), std::ios::binary);
        if (!file)
        {
            common::log_error() << "Failed to open file: " << filename.str() << std::endl;
            return;
        }

        file << "P6\n" << m_extent.width << " " << m_extent.height << "\n255\n";

        uint8_t *data = static_cast<uint8_t *>(m_device.mapMemory(m_staging_memory, 0, m_buffer_size));

        for (uint32_t i = 0; i < m_extent.width * m_extent.height; ++i)
        {
            uint8_t b = data[i * 4 + 0];
            uint8_t g = data[i * 4 + 1];
            uint8_t r = data[i * 4 + 2];

            file.write(reinterpret_cast<const char *>(&r), 1);
            file.write(reinterpret_cast<const char *>(&g), 1);
            file.write(reinterpret_cast<const char *>(&b), 1);
        }

        m_device.unmapMemory(m_staging_memory);
        file.close();
    }
};

enum class ColorMode
{
    Fixed = 0,
    Ordering = 1,
    Trace = 8,
    TraceDistance = 9,
    TraceBucket = 10,
    TraceChanged = 11
};

inline bool is_trace_mode(ColorMode mode)
{
    return (static_cast<uint32_t>(mode) & static_cast<uint32_t>(ColorMode::Trace)) != 0;
}

struct PushConstants
{
    float offset_x;
    float offset_y;
    float scale_x;
    float scale_y;
    float point_size;
};

struct SharedContext
{
    std::mutex sssp_mutex;
    std::condition_variable sssp_cv;
    bool sssp_run_requested = false;
    std::optional<uint32_t> sssp_result;
    bool sssp_shutdown = false;

    std::mutex color_mutex;
    std::condition_variable color_cv;
    bool color_update_requested = false;
    bool color_shutdown = false;
    ColorMode current_color_mode = ColorMode::Fixed;
    uint32_t delta = 3600;

    gpu::DeltaStepTracer tracer;
};

struct State
{
    double pan_x = 0.0;
    double pan_y = 0.0;
    double zoom = 1.0;
    double point_size = 3.0;

    double last_mouse_x = 0.0;
    double last_mouse_y = 0.0;
    bool is_dragging = false;

    ColorMode color_mode = ColorMode::Fixed;
    bool restart_requested = false;

    std::unique_ptr<FrameRecorder> recorder;
};

State g_state;
SharedContext *g_shared_ctx = nullptr;

void scroll_callback(GLFWwindow *, double, double yoffset)
{
    double zoom_factor = 1.1;
    if (yoffset > 0)
    {
        g_state.zoom *= zoom_factor;
    }
    else if (yoffset < 0)
    {
        g_state.zoom /= zoom_factor;
    }
}

void mouse_button_callback(GLFWwindow *window, int button, int action, int)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS)
        {
            g_state.is_dragging = true;
            glfwGetCursorPos(window, &g_state.last_mouse_x, &g_state.last_mouse_y);
        }
        else if (action == GLFW_RELEASE)
        {
            g_state.is_dragging = false;
        }
    }
}

void cursor_pos_callback(GLFWwindow *, double xpos, double ypos)
{
    if (g_state.is_dragging)
    {
        double dx = xpos - g_state.last_mouse_x;
        double dy = ypos - g_state.last_mouse_y;

        g_state.pan_x += dx / g_state.zoom;
        g_state.pan_y -= dy / g_state.zoom;

        g_state.last_mouse_x = xpos;
        g_state.last_mouse_y = ypos;
    }
}

void key_callback(GLFWwindow *, int key, int, int action, int)
{
    if (action == GLFW_PRESS || action == GLFW_REPEAT)
    {
        if (key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD)
        {
            g_state.point_size = std::min(g_state.point_size * 1.2, 100.0);
        }
        else if (key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT)
        {
            g_state.point_size = std::max(g_state.point_size / 1.2, 0.1);
        }
    }
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_M)
        {
            if (g_state.color_mode == ColorMode::Fixed)
            {
                g_state.color_mode = ColorMode::Ordering;
            }
            else
            {
                g_state.color_mode = ColorMode::Fixed;
            }
        }

        else if (key == GLFW_KEY_T)
        {
            if (g_state.color_mode == ColorMode::TraceDistance)
            {
                g_state.color_mode = ColorMode::TraceBucket;
            }
            else if (g_state.color_mode == ColorMode::TraceBucket)
            {
                g_state.color_mode = ColorMode::TraceChanged;
            }
            else
            {
                g_state.color_mode = ColorMode::TraceDistance;
            }
        }
        else if (key == GLFW_KEY_S)
        {
            if (is_trace_mode(g_state.color_mode) && g_shared_ctx)
            {
                g_shared_ctx->tracer.step();
            }
        }
        else if (key == GLFW_KEY_A)
        {
            if (is_trace_mode(g_state.color_mode) && g_shared_ctx)
            {
                if (g_shared_ctx->tracer.is_auto_playing())
                {
                    g_shared_ctx->tracer.stop_auto_play();
                }
                else
                {
                    g_shared_ctx->tracer.start_auto_play(500);
                }
            }
        }
        else if (key == GLFW_KEY_LEFT_BRACKET)
        {
            if (is_trace_mode(g_state.color_mode) && g_shared_ctx &&
                g_shared_ctx->tracer.is_auto_playing())
            {
                g_shared_ctx->tracer.set_auto_play_speed(1000);
            }
        }
        else if (key == GLFW_KEY_RIGHT_BRACKET)
        {
            if (is_trace_mode(g_state.color_mode) && g_shared_ctx &&
                g_shared_ctx->tracer.is_auto_playing())
            {
                g_shared_ctx->tracer.set_auto_play_speed(200);
            }
        }
        else if (key == GLFW_KEY_C)
        {
            if (is_trace_mode(g_state.color_mode) && g_shared_ctx)
            {
                g_shared_ctx->tracer.continue_to_end();
            }
        }
        else if (key == GLFW_KEY_R)
        {
            if (is_trace_mode(g_state.color_mode) && g_shared_ctx)
            {
                g_shared_ctx->tracer.continue_to_end();

                g_state.restart_requested = true;
            }
        }
        else if (key == GLFW_KEY_F)
        {
            if (g_state.recorder)
            {
                if (g_state.recorder->is_recording())
                {
                    g_state.recorder->stop_recording();
                }
                else
                {
                    g_state.recorder->start_recording();
                }
            }
        }
    }
}

void framebuffer_resize_callback(GLFWwindow *window, int, int)
{
    auto *context =
        reinterpret_cast<gpu::VulkanGraphicsContext *>(glfwGetWindowUserPointer(window));
    if (context)
    {
        context->set_framebuffer_resized();
    }
}

void setup_glfw_callbacks(GLFWwindow *window, gpu::VulkanGraphicsContext *context)
{
    glfwSetWindowUserPointer(window, context);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_resize_callback);
}

std::thread start_sssp_thread(SharedContext &ctx,
                              const common::WeightedGraph<uint32_t> &graph,
                              gpu::DeltaStepBuffers &deltastep_buffers,
                              vk::Device device,
                              vk::PhysicalDeviceMemoryProperties mem_props,
                              gpu::SharedQueue &queue)
{
    return std::thread(
        [&ctx, &graph, &deltastep_buffers, device, mem_props, &queue]() mutable
        {
            vk::CommandPoolCreateInfo pool_info(vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                                0);
            vk::CommandPool cmd_pool = device.createCommandPool(pool_info);

            gpu::GraphBuffers<common::WeightedGraph<uint32_t>> graph_buffers(
                graph, device, mem_props, cmd_pool, queue);
            gpu::Statistics statistics(device, mem_props);
            gpu::DeltaStep<common::WeightedGraph<uint32_t>> deltastep(
                graph_buffers, deltastep_buffers, device, statistics);
            deltastep.initialize();

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<uint32_t> dist(0, graph.num_nodes() - 1);

            while (true)
            {
                {
                    std::unique_lock<std::mutex> lock(ctx.sssp_mutex);
                    ctx.sssp_cv.wait(
                        lock, [&ctx] { return ctx.sssp_run_requested || ctx.sssp_shutdown; });

                    if (ctx.sssp_shutdown)
                    {
                        break;
                    }

                    ctx.sssp_run_requested = false;
                }

                uint32_t src_node = dist(gen);
                uint32_t dst_node = UINT32_MAX;
                uint32_t delta = ctx.delta;

                common::log() << "SSSP thread: Running delta-stepping from node " << src_node
                              << std::endl;

                uint32_t result =
                    deltastep.run(cmd_pool, queue, src_node, dst_node, delta, 1, &ctx.tracer);

                {
                    std::lock_guard<std::mutex> lock(ctx.sssp_mutex);
                    ctx.sssp_result = result;
                }

                {
                    std::lock_guard<std::mutex> lock(ctx.color_mutex);
                    ctx.color_update_requested = true;
                    ctx.color_cv.notify_one();
                }

                common::log() << "SSSP thread: Completed with result " << result << std::endl;
            }

            device.destroyCommandPool(cmd_pool);
            common::log() << "SSSP thread: Shutting down" << std::endl;
        });
}

std::thread start_color_updater_thread(SharedContext &ctx,
                                       gpu::DeltaStepBuffers &deltastep_buffers,
                                       vk::Device device,
                                       vk::Buffer color_buffer,
                                       size_t num_nodes,
                                       gpu::SharedQueue &queue)
{
    return std::thread(
        [&ctx, &deltastep_buffers, device, color_buffer, num_nodes, &queue]() mutable
        {
            vk::CommandPoolCreateInfo pool_info(vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                                0);
            vk::CommandPool cmd_pool = device.createCommandPool(pool_info);

            struct NodeColorPushConsts
            {
                uint32_t num_nodes;
                uint32_t max_distance;
                uint32_t color_mode;
                uint32_t delta;
                uint32_t bucket_index;
                uint32_t buffer_index;
            };

            auto deltastep_bufs = deltastep_buffers.buffers();
            auto node_color_pipeline = gpu::create_compute_pipeline<NodeColorPushConsts>(
                device,
                "node_color.spv",
                {{deltastep_bufs[0], color_buffer, deltastep_bufs[2], deltastep_bufs[3]}});

            auto dispatch_shader = [&](const uint32_t color_mode_value,
                                       const uint32_t max_distance,
                                       const uint32_t delta,
                                       const std::optional<gpu::DeltaStepPayload> &maybe_payload)
            {
                auto bucket_index = UINT32_MAX;
                auto buffer_index = UINT32_MAX;
                if (maybe_payload)
                {
                    bucket_index = maybe_payload->bucket_index;
                    buffer_index = maybe_payload->buffer_index;
                }
                NodeColorPushConsts pc{static_cast<uint32_t>(num_nodes),
                                       max_distance,
                                       color_mode_value,
                                       delta,
                                       bucket_index,
                                       buffer_index};

                vk::CommandBuffer cmd = device.allocateCommandBuffers(
                    {cmd_pool, vk::CommandBufferLevel::ePrimary, 1})[0];

                cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
                cmd.bindPipeline(vk::PipelineBindPoint::eCompute, node_color_pipeline.pipeline);
                cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                       node_color_pipeline.layout,
                                       0,
                                       1,
                                       &node_color_pipeline.descriptor_sets[0],
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
                queue.submit({{0, nullptr, nullptr, 1, &cmd}}, fence);
                (void)device.waitForFences(1, &fence, VK_TRUE, UINT64_MAX);

                device.destroyFence(fence);
                device.freeCommandBuffers(cmd_pool, 1, &cmd);
            };

            while (!ctx.color_shutdown)
            {
                ColorMode mode;
                {
                    std::unique_lock<std::mutex> lock(ctx.color_mutex);
                    ctx.color_cv.wait(
                        lock, [&ctx] { return ctx.color_update_requested || ctx.color_shutdown; });

                    if (ctx.color_shutdown)
                    {
                        break;
                    }

                    ctx.color_update_requested = false;
                    mode = ctx.current_color_mode;
                }

                if (is_trace_mode(mode))
                {
                    while (true)
                    {
                        mode = ctx.current_color_mode;

                        if (!is_trace_mode(mode))
                        {
                            break;
                        }

                        if (ctx.tracer.is_finished())
                        {
                            dispatch_shader(static_cast<uint32_t>(mode),
                                            *deltastep_buffers.max_distance(),
                                            ctx.delta,
                                            ctx.tracer.payload());
                            break;
                        }

                        if (ctx.tracer.wait_for_signal(100))
                        {
                            dispatch_shader(static_cast<uint32_t>(mode),
                                            *deltastep_buffers.max_distance(),
                                            ctx.delta,
                                            ctx.tracer.payload());
                        }
                    }
                }
                else if (mode == ColorMode::Fixed)
                {
                    dispatch_shader(static_cast<uint32_t>(mode), 0, 0, {});
                }
                else if (mode == ColorMode::Ordering)
                {
                    dispatch_shader(static_cast<uint32_t>(mode), 0, 0, {});
                }
            }

            device.destroyPipeline(node_color_pipeline.pipeline);
            device.destroyPipelineLayout(node_color_pipeline.layout);
            device.destroyDescriptorSetLayout(node_color_pipeline.descriptor_set_layout);
            device.destroyDescriptorPool(node_color_pipeline.descriptor_pool);
            device.destroyShaderModule(node_color_pipeline.shader);
            device.destroyCommandPool(cmd_pool);
            common::log() << "Color updater thread: Shutting down" << std::endl;
        });
}

std::pair<vk::Buffer, vk::DeviceMemory> create_color_buffer(
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

void project_coordinates(gpu::CoordinatesBuffer &coord_buffer,
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

    float center_x = (min_point.x + max_point.x) / 2.0;
    float center_y = (min_point.y + max_point.y) / 2.0;
    float inv_width = 1.0 / (max_point.x - min_point.x);
    float inv_height = 1.0 / (max_point.y - min_point.y);

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
                                   &compute_pipeline.descriptor_sets[0],
                                   0,
                                   nullptr);

    ProjectionPushConstants push_constants{
        center_x, center_y, inv_width, inv_height, static_cast<uint32_t>(num_coords)};

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
    device.destroyPipeline(compute_pipeline.pipeline);
    device.destroyPipelineLayout(compute_pipeline.layout);
    device.destroyDescriptorSetLayout(compute_pipeline.descriptor_set_layout);
    device.destroyDescriptorPool(compute_pipeline.descriptor_pool);
    device.destroyShaderModule(compute_pipeline.shader);
}

std::pair<vk::Pipeline, vk::PipelineLayout> create_graphics_pipeline(vk::Device device,
                                                                     vk::RenderPass render_pass,
                                                                     vk::Extent2D swapchain_extent)
{
    vk::ShaderModule vert_shader = gpu::create_shader_module(device, "visualize_node.vert.spv");
    vk::ShaderModule frag_shader = gpu::create_shader_module(device, "visualize_node.frag.spv");

    vk::PipelineShaderStageCreateInfo vert_stage_info(
        {}, vk::ShaderStageFlagBits::eVertex, vert_shader, "main");
    vk::PipelineShaderStageCreateInfo frag_stage_info(
        {}, vk::ShaderStageFlagBits::eFragment, frag_shader, "main");

    vk::PipelineShaderStageCreateInfo shader_stages[] = {vert_stage_info, frag_stage_info};

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
                                                 2,
                                                 shader_stages,
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

PushConstants calculate_view_transform(const State &camera,
                                       vk::Extent2D swapchain_extent,
                                       double graph_width,
                                       double graph_height)
{
    double aspect_ratio = static_cast<double>(swapchain_extent.width) / swapchain_extent.height;
    double graph_aspect = graph_width / graph_height;

    double scale_x, scale_y;
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

    return PushConstants{static_cast<float>(normalized_pan_x),
                         static_cast<float>(normalized_pan_y),
                         static_cast<float>(scale_x),
                         static_cast<float>(scale_y),
                         std::max(0.5f, static_cast<float>(camera.point_size * camera.zoom))};
}

void record_render_commands(vk::CommandBuffer command_buffer,
                            vk::RenderPass render_pass,
                            vk::Framebuffer framebuffer,
                            vk::Extent2D swapchain_extent,
                            vk::Pipeline graphics_pipeline,
                            vk::PipelineLayout pipeline_layout,
                            const PushConstants &push_constants,
                            vk::Buffer projected_coords_buffer,
                            vk::Buffer color_buffer,
                            uint32_t vertex_count)
{
    command_buffer.reset();

    vk::CommandBufferBeginInfo begin_info;
    command_buffer.begin(begin_info);

    vk::ClearValue clear_color(vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}));

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

    vk::Buffer vertex_buffers[] = {projected_coords_buffer, color_buffer};
    vk::DeviceSize offsets[] = {0, 0};
    command_buffer.bindVertexBuffers(0, 2, vertex_buffers, offsets);

    command_buffer.draw(vertex_count, 1, 0, 0);

    command_buffer.endRenderPass();
    command_buffer.end();
}

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3)
    {
        common::log_error() << "Usage: " << argv[0] << " <graph_base_path> [output_dir]" << std::endl;
        return EXIT_FAILURE;
    }

    std::string base_path = argv[1];
    std::optional<std::string> output_dir;
    if (argc == 3)
    {
        output_dir = argv[2];
    }

    common::log() << "Loading graph from: " << base_path << std::endl;

    auto graph = common::files::read_weighted_graph<uint32_t>(base_path);
    auto coordinates = common::files::read_coordinates(base_path);

    if (graph.num_nodes() != coordinates.size())
    {
        common::log_error() << "Graph node count (" << graph.num_nodes()
                            << ") does not match coordinate count (" << coordinates.size() << ")"
                            << std::endl;
        return EXIT_FAILURE;
    }

    common::log() << "Loaded graph with " << graph.num_nodes() << " nodes and " << graph.num_edges()
                  << " edges" << std::endl;

    auto bounding_box = common::bounds(coordinates);
    auto min_point = common::to_web_mercator(bounding_box.south_east);
    auto max_point = common::to_web_mercator(bounding_box.north_west);

    common::log() << "Creating Vulkan graphics context..." << std::endl;

    gpu::VulkanGraphicsContext context("Graph Visualizer", 1280, 720);

    setup_glfw_callbacks(context.window(), &context);

    vk::Device device = context.device();
    vk::PhysicalDeviceMemoryProperties mem_props = context.physical_device().getMemoryProperties();
    vk::CommandPool cmd_pool = context.command_pool();
    gpu::SharedQueue &queue = context.shared_queue();

    gpu::CoordinatesBuffer coord_buffer(coordinates, device, mem_props, cmd_pool, queue);

    auto [color_buffer, color_buffer_memory] =
        create_color_buffer(device, mem_props, coordinates.size());

    project_coordinates(
        coord_buffer, device, cmd_pool, queue, min_point, max_point, coordinates.size());

    auto [graphics_pipeline, pipeline_layout] =
        create_graphics_pipeline(device, context.render_pass(), context.swapchain_extent());

    constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;
    vk::CommandBufferAllocateInfo alloc_info(
        cmd_pool, vk::CommandBufferLevel::ePrimary, MAX_FRAMES_IN_FLIGHT);

    std::vector<vk::CommandBuffer> command_buffers = device.allocateCommandBuffers(alloc_info);

    common::log() << "Creating shared DeltaStepBuffers..." << std::endl;
    gpu::DeltaStepBuffers deltastep_buffers(graph.num_nodes(), device, mem_props);

    common::log() << "Creating shared context..." << std::endl;
    SharedContext shared_ctx;
    g_shared_ctx = &shared_ctx;

    common::log() << "Starting worker threads..." << std::endl;
    auto sssp_thread =
        start_sssp_thread(shared_ctx, graph, deltastep_buffers, device, mem_props, queue);
    auto color_thread = start_color_updater_thread(
        shared_ctx, deltastep_buffers, device, color_buffer, coordinates.size(), queue);

    {
        std::lock_guard<std::mutex> lock(shared_ctx.color_mutex);
        shared_ctx.color_update_requested = true;
        shared_ctx.color_cv.notify_one();
    }

    if (output_dir)
    {
        common::log() << "Frame recording enabled. Press 'F' to start/stop recording." << std::endl;
        g_state.recorder = std::make_unique<FrameRecorder>(device,
                                                            mem_props,
                                                            cmd_pool,
                                                            queue,
                                                            context.swapchain_extent(),
                                                            context.swapchain_format(),
                                                            *output_dir);
    }

    common::log() << "Starting render loop..." << std::endl;

    g_state.zoom = 1.0;
    g_state.pan_x = 0.0;
    g_state.pan_y = 0.0;

    auto coord_buffers = coord_buffer.buffers();

    ColorMode previous_mode = g_state.color_mode;
    uint32_t current_frame = 0;

    while (!context.should_close())
    {
        context.poll_events();

        if (g_state.restart_requested)
        {
            g_state.restart_requested = false;
            previous_mode = ColorMode::Fixed;
            g_state.color_mode = ColorMode::TraceDistance;
        }

        if (g_state.color_mode != previous_mode)
        {
            if (is_trace_mode(g_state.color_mode))
            {
                common::log() << "Entering Trace mode" << std::endl;

                {
                    std::lock_guard<std::mutex> lock(shared_ctx.sssp_mutex);
                    shared_ctx.sssp_run_requested = true;
                    shared_ctx.sssp_cv.notify_one();
                }
            }

            {
                std::lock_guard<std::mutex> lock(shared_ctx.color_mutex);
                shared_ctx.current_color_mode = g_state.color_mode;
                shared_ctx.color_update_requested = true;
                shared_ctx.color_cv.notify_one();
            }

            previous_mode = g_state.color_mode;
        }

        auto frame = context.begin_frame();

        PushConstants push_constants = calculate_view_transform(g_state,
                                                                context.swapchain_extent(),
                                                                max_point.x - min_point.x,
                                                                max_point.y - min_point.y);

        vk::CommandBuffer command_buffer = command_buffers[current_frame];

        record_render_commands(command_buffer,
                               context.render_pass(),
                               frame.framebuffer,
                               context.swapchain_extent(),
                               graphics_pipeline,
                               pipeline_layout,
                               push_constants,
                               coord_buffers[1],
                               color_buffer,
                               static_cast<uint32_t>(coordinates.size()));

        vk::PipelineStageFlags wait_stages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
        vk::SubmitInfo submit_info(
            1, &frame.image_available, wait_stages, 1, &command_buffer, 1, &frame.render_finished);

        context.shared_queue().submit(1, &submit_info, frame.in_flight);

        context.end_frame(frame);

        if (g_state.recorder && g_state.recorder->is_recording())
        {
            device.waitIdle();
            g_state.recorder->capture_frame(frame.image);
        }

        current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    device.waitIdle();

    common::log() << "Shutting down worker threads..." << std::endl;

    {
        std::lock_guard<std::mutex> lock(shared_ctx.sssp_mutex);
        shared_ctx.sssp_shutdown = true;
        shared_ctx.sssp_cv.notify_one();
    }
    {
        std::lock_guard<std::mutex> lock(shared_ctx.color_mutex);
        shared_ctx.color_shutdown = true;
        shared_ctx.color_cv.notify_one();
    }

    shared_ctx.tracer.continue_to_end();

    if (sssp_thread.joinable())
    {
        sssp_thread.join();
    }
    if (color_thread.joinable())
    {
        color_thread.join();
    }

    g_shared_ctx = nullptr;

    device.destroyPipeline(graphics_pipeline);
    device.destroyPipelineLayout(pipeline_layout);
    device.destroyBuffer(color_buffer);
    device.freeMemory(color_buffer_memory);

    common::log() << "Visualization complete." << std::endl;

    return EXIT_SUCCESS;
}
