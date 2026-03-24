#ifndef GPUSSSP_GPU_BELLMANFORD_HPP
#define GPUSSSP_GPU_BELLMANFORD_HPP

#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.hpp>

#include "gpu/bellmanford_buffers.hpp"
#include "gpu/graph_buffers.hpp"
#include "gpu/shader.hpp"
#include "gpu/statistics.hpp"

namespace gpusssp::gpu
{

template <typename GraphT> class BellmanFord
{
    static constexpr const size_t DEFAULT_WORKGROUP_SIZE = 128u;
    static constexpr const uint32_t BATCH_SIZE = 32u;

    struct PushConsts
    {
        uint32_t n;
    };

  public:
    BellmanFord(const GraphBuffers<GraphT> &graph_buffers,
                BellmanFordBuffers &bellmanford_buffers,
                vk::Device device,
                Statistics &statistics,
                uint32_t workgroup_size = DEFAULT_WORKGROUP_SIZE)
        : graph_buffers(graph_buffers), bellmanford_buffers(bellmanford_buffers),
          statistics(statistics), device(device), workgroup_size(workgroup_size)
    {
        auto [first_edges_buffer, targets_buffer, weights_buffer] = graph_buffers.buffers();
        auto dist_buffer = bellmanford_buffers.dist_buffer();
        auto changed_buffer = bellmanford_buffers.changed_buffer();
        auto statistics_buffer = statistics.buffer();

        main_pipeline = create_compute_pipeline<PushConsts>(device,
                                                            "bellman_ford.spv",
                                                            {{first_edges_buffer,
                                                              targets_buffer,
                                                              weights_buffer,
                                                              dist_buffer,
                                                              changed_buffer,
                                                              statistics_buffer}},
                                                            {workgroup_size});

        fence = device.createFence({vk::FenceCreateFlagBits::eSignaled});
    }

    ~BellmanFord()
    {
        main_pipeline.destroy(device);
        device.destroyFence(fence);
    }

    void initialize(vk::CommandPool cmd_pool)
    {
        device.freeCommandBuffers(cmd_pool, batch_cmd_bufs);

        batch_cmd_bufs =
            device.allocateCommandBuffers({cmd_pool, vk::CommandBufferLevel::ePrimary, 1});

        auto num_nodes = static_cast<uint32_t>(graph_buffers.num_nodes());
        record_batch_commands(batch_cmd_bufs[0], num_nodes);
    }

    void record_batch_commands(vk::CommandBuffer cmd_buf, uint32_t num_nodes)
    {
        cmd_buf.begin({vk::CommandBufferUsageFlagBits::eSimultaneousUse});
        cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, main_pipeline.pipeline);
        cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                   main_pipeline.layout,
                                   0,
                                   main_pipeline.descriptor_sets[0],
                                   {});

        PushConsts pc{num_nodes};
        cmd_buf.pushConstants(
            main_pipeline.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);

        for (uint32_t i = 0; i < BATCH_SIZE; ++i)
        {
            bellmanford_buffers.cmd_clear_changed(cmd_buf);
            cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                    vk::PipelineStageFlagBits::eComputeShader,
                                    vk::DependencyFlags{},
                                    vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite,
                                                      vk::AccessFlagBits::eShaderRead |
                                                          vk::AccessFlagBits::eShaderWrite},
                                    {},
                                    {});

            cmd_buf.dispatch((num_nodes + workgroup_size - 1) / workgroup_size, 1, 1);
            cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                    vk::PipelineStageFlagBits::eComputeShader |
                                        vk::PipelineStageFlagBits::eTransfer,
                                    vk::DependencyFlags{},
                                    vk::MemoryBarrier{vk::AccessFlagBits::eShaderWrite,
                                                      vk::AccessFlagBits::eShaderRead |
                                                          vk::AccessFlagBits::eShaderWrite |
                                                          vk::AccessFlagBits::eTransferRead},
                                    {},
                                    {});
        }

        bellmanford_buffers.cmd_sync_changed(cmd_buf);
        cmd_buf.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eHost,
            vk::DependencyFlags{},
            vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead},
            {},
            {});

        cmd_buf.end();
    }

    void record_init_commands(vk::CommandBuffer cmd_buf, uint32_t src_node)
    {
        cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        bellmanford_buffers.cmd_init_dist(cmd_buf, src_node);

        cmd_buf.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlags{},
            vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite,
                              vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite},
            {},
            {});

        cmd_buf.end();
    }

    void record_sync_commands(vk::CommandBuffer cmd_buf, uint32_t dst_node)
    {
        cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        cmd_buf.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eTransfer,
            vk::DependencyFlags{},
            vk::MemoryBarrier{vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead},
            {},
            {});

        bellmanford_buffers.cmd_sync_dist(cmd_buf, dst_node);

        cmd_buf.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eHost,
            vk::DependencyFlags{},
            vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead},
            {},
            {});

        cmd_buf.end();
    }

    template <typename QueueT>
    uint32_t run(vk::CommandPool cmd_pool, QueueT queue, uint32_t src_node, uint32_t dst_node)
    {
        if (batch_cmd_bufs.empty())
        {
            throw std::runtime_error(
                "BellmanFord was not initialized, the command buffers are empty.");
        }

        auto num_nodes = static_cast<uint32_t>(graph_buffers.num_nodes());

        std::vector<vk::CommandBuffer> one_time_cmd_bufs =
            device.allocateCommandBuffers({cmd_pool, vk::CommandBufferLevel::ePrimary, 2});
        auto &init_cmd_buf = one_time_cmd_bufs[0];
        auto &sync_cmd_buf = one_time_cmd_bufs[1];

        record_init_commands(init_cmd_buf, src_node);
        record_sync_commands(sync_cmd_buf, dst_node);

        device.resetFences(fence);
        queue.submit(vk::SubmitInfo{0, nullptr, nullptr, 1, &init_cmd_buf}, fence);
        (void)device.waitForFences(fence, VK_TRUE, UINT64_MAX);

        uint32_t *gpu_changed = bellmanford_buffers.changed();
        uint32_t *gpu_best_distance = bellmanford_buffers.best_distance();

        for (uint32_t iteration = 0; iteration < num_nodes - 1; iteration += BATCH_SIZE)
        {
            device.resetFences(fence);
            queue.submit(vk::SubmitInfo{0, nullptr, nullptr, 1, batch_cmd_bufs.data()}, fence);
            (void)device.waitForFences(fence, VK_TRUE, UINT64_MAX);

            if (*gpu_changed == 0)
            {
                break;
            }
        }

        device.resetFences(fence);
        queue.submit(vk::SubmitInfo{0, nullptr, nullptr, 1, &sync_cmd_buf}, fence);
        (void)device.waitForFences(fence, VK_TRUE, UINT64_MAX);

        device.freeCommandBuffers(cmd_pool, one_time_cmd_bufs);

        return *gpu_best_distance;
    }

  private:
    const GraphBuffers<GraphT> &graph_buffers;
    BellmanFordBuffers &bellmanford_buffers;
    Statistics &statistics;

    ComputePipeline main_pipeline;

    vk::Device device;
    vk::Fence fence;
    uint32_t workgroup_size;
    std::vector<vk::CommandBuffer> batch_cmd_bufs;
};

} // namespace gpusssp::gpu

#endif
