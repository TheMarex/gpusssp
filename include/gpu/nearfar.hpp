#ifndef GPUSSSP_GPU_NEARFAR_HPP
#define GPUSSSP_GPU_NEARFAR_HPP

#include <cstddef>
#include <cstdint>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>

#include "common/constants.hpp"
#include "common/logger.hpp"
#include "common/statistics.hpp"
#include "gpu/graph_buffers.hpp"
#include "gpu/nearfar_buffers.hpp"
#include "gpu/shader.hpp"
#include "gpu/statistics.hpp"

namespace gpusssp::gpu
{

template <typename GraphT> class NearFar
{
    static constexpr const size_t DEFAULT_WORKGROUP_SIZE = 64u;
    static constexpr const uint32_t DEFAULT_RELAX_BATCH_SIZE = 64u;
    struct PushConsts
    {
        uint32_t n;
    };

  public:
    NearFar(const GraphBuffers<GraphT> &graph_buffers,
            NearFarBuffers &nearfar_buffers,
            vk::Device device,
            Statistics &statistics,
            uint32_t delta,
            uint32_t relax_batch_size = DEFAULT_RELAX_BATCH_SIZE,
            uint32_t workgroup_size = DEFAULT_WORKGROUP_SIZE)
        : graph_buffers(graph_buffers), nearfar_buffers(nearfar_buffers), statistics(statistics),
          device(device), delta(delta), relax_batch_size(relax_batch_size),
          workgroup_size(workgroup_size)
    {
        auto [first_edges_buffer, targets_buffer, weights_buffer] = graph_buffers.buffers();
        auto dist_buffer = nearfar_buffers.dist_buffer();
        auto [near_0_buffer, near_1_buffer] = nearfar_buffers.near_buffers();
        auto [far_0_buffer, far_1_buffer] = nearfar_buffers.far_buffers();
        auto dispatch_buffer = nearfar_buffers.dispatch_buffer();
        auto processed_buffer = nearfar_buffers.processed_buffer();
        auto phase_params_buffer = nearfar_buffers.phase_params_buffer();
        auto statistics_buffer = statistics.buffer();

        relax_pipeline = create_compute_pipeline<PushConsts>(device,
                                                             "nearfar_relax.spv",
                                                             {{first_edges_buffer,
                                                               targets_buffer,
                                                               weights_buffer,
                                                               dist_buffer,
                                                               near_0_buffer,
                                                               near_1_buffer,
                                                               far_0_buffer,
                                                               phase_params_buffer,
                                                               statistics_buffer},
                                                              {first_edges_buffer,
                                                               targets_buffer,
                                                               weights_buffer,
                                                               dist_buffer,
                                                               near_1_buffer,
                                                               near_0_buffer,
                                                               far_0_buffer,
                                                               phase_params_buffer,
                                                               statistics_buffer},
                                                              {first_edges_buffer,
                                                               targets_buffer,
                                                               weights_buffer,
                                                               dist_buffer,
                                                               near_0_buffer,
                                                               near_1_buffer,
                                                               far_1_buffer,
                                                               phase_params_buffer,
                                                               statistics_buffer},
                                                              {first_edges_buffer,
                                                               targets_buffer,
                                                               weights_buffer,
                                                               dist_buffer,
                                                               near_1_buffer,
                                                               near_0_buffer,
                                                               far_1_buffer,
                                                               phase_params_buffer,
                                                               statistics_buffer}},
                                                             {workgroup_size});

        compact_pipeline = create_compute_pipeline<PushConsts>(device,
                                                               "nearfar_compact.spv",
                                                               {
                                                                   {dist_buffer,
                                                                    far_0_buffer,
                                                                    near_0_buffer,
                                                                    far_1_buffer,
                                                                    processed_buffer,
                                                                    phase_params_buffer,
                                                                    statistics_buffer},
                                                                   {dist_buffer,
                                                                    far_1_buffer,
                                                                    near_0_buffer,
                                                                    far_0_buffer,
                                                                    processed_buffer,
                                                                    phase_params_buffer,
                                                                    statistics_buffer},
                                                               },
                                                               {workgroup_size});

        prepare_dispatch_pipeline =
            create_compute_pipeline<uint32_t>(device,
                                              "nearfar_prepare_dispatch.spv",
                                              {{near_0_buffer, dispatch_buffer},
                                               {near_1_buffer, dispatch_buffer},
                                               {far_0_buffer, dispatch_buffer},
                                               {far_1_buffer, dispatch_buffer}},
                                              {workgroup_size});
    }

    ~NearFar()
    {
        relax_pipeline.destroy(device);
        compact_pipeline.destroy(device);
        prepare_dispatch_pipeline.destroy(device);
    }

    void initialize(vk::CommandPool cmd_pool)
    {
        device.freeCommandBuffers(cmd_pool, relax_cmd_bufs);
        device.freeCommandBuffers(cmd_pool, compact_cmd_bufs);

        relax_cmd_bufs =
            device.allocateCommandBuffers({cmd_pool, vk::CommandBufferLevel::ePrimary, 2});
        compact_cmd_bufs =
            device.allocateCommandBuffers({cmd_pool, vk::CommandBufferLevel::ePrimary, 2});

        auto num_nodes = static_cast<uint32_t>(graph_buffers.num_nodes());
        auto dispatch_buffer = nearfar_buffers.dispatch_buffer();

        for (auto idx = 0u; idx < 2; ++idx)
        {
            record_relax_batch_commands(relax_cmd_bufs[idx], num_nodes, idx, dispatch_buffer);
            record_compact_commands(compact_cmd_bufs[idx], num_nodes, idx, dispatch_buffer);
        }
    }

    void record_relax_batch_commands(vk::CommandBuffer cmd_buf,
                                     uint32_t num_nodes,
                                     uint32_t current_far_buffer_idx,
                                     vk::Buffer dispatch_buffer)
    {
        auto record_start =
            common::Statistics::start(common::StatisticsEvent::NEARFAR_CMDBUF_RECORD_DURATION);

        cmd_buf.begin({vk::CommandBufferUsageFlagBits::eSimultaneousUse});

        for (uint32_t batch_iter = 0; batch_iter < relax_batch_size; ++batch_iter)
        {
            uint32_t current_near_buffer_idx = batch_iter % 2;
            auto relax_desc_set =
                relax_pipeline
                    .descriptor_sets[current_near_buffer_idx + (current_far_buffer_idx * 2)];

            nearfar_buffers.cmd_clear_near(cmd_buf, 1 - current_near_buffer_idx);
            cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                    vk::PipelineStageFlagBits::eComputeShader,
                                    vk::DependencyFlags{},
                                    vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite,
                                                      vk::AccessFlagBits::eShaderRead |
                                                          vk::AccessFlagBits::eShaderWrite},
                                    {},
                                    {});

            cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute,
                                 prepare_dispatch_pipeline.pipeline);
            cmd_buf.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute,
                prepare_dispatch_pipeline.layout,
                0,
                prepare_dispatch_pipeline.descriptor_sets[current_near_buffer_idx],
                {});
            cmd_buf.pushConstants(prepare_dispatch_pipeline.layout,
                                  vk::ShaderStageFlagBits::eCompute,
                                  0,
                                  4,
                                  &num_nodes);
            cmd_buf.dispatch(1, 1, 1);

            cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                    vk::PipelineStageFlagBits::eComputeShader,
                                    vk::DependencyFlags{},
                                    vk::MemoryBarrier{vk::AccessFlagBits::eShaderWrite,
                                                      vk::AccessFlagBits::eShaderRead |
                                                          vk::AccessFlagBits::eIndirectCommandRead},
                                    {},
                                    {});

            cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, relax_pipeline.pipeline);
            cmd_buf.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, relax_pipeline.layout, 0, relax_desc_set, {});

            PushConsts pc{num_nodes};
            cmd_buf.pushConstants(
                relax_pipeline.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
            cmd_buf.dispatchIndirect(dispatch_buffer, 0);

            cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                    vk::PipelineStageFlagBits::eTransfer |
                                        vk::PipelineStageFlagBits::eComputeShader,
                                    vk::DependencyFlags{},
                                    vk::MemoryBarrier{vk::AccessFlagBits::eShaderWrite,
                                                      vk::AccessFlagBits::eTransferRead |
                                                          vk::AccessFlagBits::eShaderRead},
                                    {},
                                    {});
        }

        nearfar_buffers.cmd_sync_near_count(cmd_buf, relax_batch_size % 2);

        cmd_buf.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlags{},
            vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite,
                              vk::AccessFlagBits::eHostRead | vk::AccessFlagBits::eShaderRead},
            {},
            {});

        cmd_buf.end();

        common::Statistics::get().stop(common::StatisticsEvent::NEARFAR_CMDBUF_RECORD_DURATION,
                                       record_start);
    }

    void record_compact_commands(vk::CommandBuffer cmd_buf,
                                 uint32_t num_nodes,
                                 uint32_t current_far_buffer_idx,
                                 vk::Buffer dispatch_buffer)
    {
        auto record_start =
            common::Statistics::start(common::StatisticsEvent::NEARFAR_CMDBUF_RECORD_DURATION);

        cmd_buf.begin({vk::CommandBufferUsageFlagBits::eSimultaneousUse});

        auto compact_desc_set = compact_pipeline.descriptor_sets[current_far_buffer_idx];

        nearfar_buffers.cmd_clear_far_and_processed(cmd_buf, current_far_buffer_idx);
        cmd_buf.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlags{},
            vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite,
                              vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite},
            {},
            {});

        cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, prepare_dispatch_pipeline.pipeline);
        cmd_buf.bindDescriptorSets(
            vk::PipelineBindPoint::eCompute,
            prepare_dispatch_pipeline.layout,
            0,
            prepare_dispatch_pipeline.descriptor_sets[current_far_buffer_idx + 2],
            {});
        cmd_buf.pushConstants(
            prepare_dispatch_pipeline.layout, vk::ShaderStageFlagBits::eCompute, 0, 4, &num_nodes);
        cmd_buf.dispatch(1, 1, 1);

        cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader,
                                vk::DependencyFlags{},
                                vk::MemoryBarrier{vk::AccessFlagBits::eShaderWrite,
                                                  vk::AccessFlagBits::eShaderRead |
                                                      vk::AccessFlagBits::eIndirectCommandRead},
                                {},
                                {});

        cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, compact_pipeline.pipeline);
        cmd_buf.bindDescriptorSets(
            vk::PipelineBindPoint::eCompute, compact_pipeline.layout, 0, compact_desc_set, {});

        PushConsts pc{num_nodes};
        cmd_buf.pushConstants(
            compact_pipeline.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
        cmd_buf.dispatchIndirect(dispatch_buffer, 0);

        cmd_buf.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eTransfer,
            vk::DependencyFlags{},
            vk::MemoryBarrier{vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead},
            {},
            {});

        nearfar_buffers.cmd_sync_near_count(cmd_buf, 0);
        nearfar_buffers.cmd_sync_far_count(cmd_buf, 1 - current_far_buffer_idx);
        nearfar_buffers.cmd_sync_phase_params(cmd_buf);

        cmd_buf.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eHost,
            vk::DependencyFlags{},
            vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead},
            {},
            {});

        cmd_buf.end();

        common::Statistics::get().stop(common::StatisticsEvent::NEARFAR_CMDBUF_RECORD_DURATION,
                                       record_start);
    }

    void record_sync_commands(vk::CommandBuffer cmd_buf,
                              uint32_t dst_node,
                              uint32_t current_far_buffer_idx)
    {
        auto record_start =
            common::Statistics::start(common::StatisticsEvent::NEARFAR_CMDBUF_RECORD_DURATION);

        cmd_buf.begin({vk::CommandBufferUsageFlagBits::eSimultaneousUse});

        cmd_buf.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eTransfer,
            vk::DependencyFlags{},
            vk::MemoryBarrier{vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead},
            {},
            {});

        nearfar_buffers.cmd_sync_dist(cmd_buf, dst_node);
        nearfar_buffers.cmd_sync_far_count(cmd_buf, current_far_buffer_idx);

        cmd_buf.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlags{},
            vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead},
            {},
            {});

        cmd_buf.end();

        common::Statistics::get().stop(common::StatisticsEvent::NEARFAR_CMDBUF_RECORD_DURATION,
                                       record_start);
    }

    void record_init_commands(vk::CommandBuffer cmd_buf,
                              uint32_t src_node,
                              uint32_t dst_node,
                              uint32_t delta)
    {
        auto record_0_start =
            common::Statistics::start(common::StatisticsEvent::NEARFAR_CMDBUF_RECORD_DURATION);
        cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        nearfar_buffers.cmd_init_dist(cmd_buf, src_node);
        nearfar_buffers.cmd_init_near_far(cmd_buf, src_node);
        nearfar_buffers.cmd_init_phase_params(cmd_buf, delta);

        cmd_buf.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
            vk::DependencyFlags{},
            vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite,
                              vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                                  vk::AccessFlagBits::eTransferRead},
            {},
            {});

        nearfar_buffers.cmd_sync_dist(cmd_buf, dst_node);
        nearfar_buffers.cmd_sync_near_count(cmd_buf, 0);
        nearfar_buffers.cmd_sync_far_count(cmd_buf, 0);
        nearfar_buffers.cmd_sync_phase_params(cmd_buf);

        cmd_buf.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eHost,
            vk::DependencyFlags{},
            vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead},
            {},
            {});

        cmd_buf.end();
        common::Statistics::get().stop(common::StatisticsEvent::NEARFAR_CMDBUF_RECORD_DURATION,
                                       record_0_start);
    }

    template <typename QueueT>
    uint32_t run(vk::CommandPool cmd_pool, QueueT queue, uint32_t src_node, uint32_t dst_node)
    {
        if (relax_cmd_bufs.empty() || compact_cmd_bufs.empty())
        {
            throw std::runtime_error("NearFar was not initialized, the command buffers are empty.");
        }
        auto init_start = common::Statistics::start(common::StatisticsEvent::NEARFAR_INIT_DURATION);

        std::vector<vk::CommandBuffer> cmd_bufs =
            device.allocateCommandBuffers({cmd_pool, vk::CommandBufferLevel::ePrimary, 3});
        auto &init_cmd_buf = cmd_bufs[0];
        auto sync_cmd_bufs = std::span(cmd_bufs.data() + 1, 2);

        uint32_t *gpu_best_distance = nearfar_buffers.best_distance();
        uint32_t *gpu_num_near = nearfar_buffers.num_near();
        uint32_t *gpu_num_far = nearfar_buffers.num_far();
        uint32_t *gpu_phase = nearfar_buffers.gpu_phase();
        uint32_t *gpu_delta = nearfar_buffers.gpu_delta();

        record_sync_commands(sync_cmd_bufs[0], dst_node, 0);
        record_sync_commands(sync_cmd_bufs[1], dst_node, 1);
        record_init_commands(init_cmd_buf, src_node, dst_node, delta);

        queue.submit(vk::SubmitInfo{0, nullptr, nullptr, 1, &init_cmd_buf});
        queue.waitIdle();

        common::Statistics::get().stop(common::StatisticsEvent::NEARFAR_INIT_DURATION, init_start);

        uint32_t current_far_buffer_idx = 0;

        while (true)
        {
            common::Statistics::get().count(common::StatisticsEvent::NEARFAR_PHASE);

            auto relax_start =
                common::Statistics::start(common::StatisticsEvent::NEARFAR_RELAX_DURATION);

            while (*gpu_num_near > 0)
            {
                common::Statistics::get().count(common::StatisticsEvent::NEARFAR_RELAX);
                common::log_debug() << *gpu_phase << " " << *gpu_num_near << " best distance "
                                    << *gpu_best_distance << '\n';

                queue.submit(vk::SubmitInfo{
                    0, nullptr, nullptr, 1, &relax_cmd_bufs[current_far_buffer_idx]});
                queue.waitIdle();
            }

            common::Statistics::get().stop(common::StatisticsEvent::NEARFAR_RELAX_DURATION,
                                           relax_start);

            queue.submit(
                vk::SubmitInfo{0, nullptr, nullptr, 1, &sync_cmd_bufs[current_far_buffer_idx]});
            queue.waitIdle();

            if (*gpu_best_distance != common::INF_WEIGHT)
            {
                if (*gpu_best_distance < *gpu_phase * *gpu_delta)
                {
                    break;
                }
            }

            auto compact_start =
                common::Statistics::start(common::StatisticsEvent::NEARFAR_COMPACT_DURATION);

            if (*gpu_num_far == 0)
            {
                break;
            }

            common::log_debug() << *gpu_phase << " far " << *gpu_num_far << " best distance "
                                << *gpu_best_distance << '\n';

            queue.submit(
                vk::SubmitInfo{0, nullptr, nullptr, 1, &compact_cmd_bufs[current_far_buffer_idx]});
            queue.waitIdle();

            current_far_buffer_idx = 1 - current_far_buffer_idx;

            common::log_debug() << *gpu_phase << " compacted to: far " << *gpu_num_far << " near "
                                << *gpu_num_near << '\n';

            common::Statistics::get().stop(common::StatisticsEvent::NEARFAR_COMPACT_DURATION,
                                           compact_start);
        }

        device.freeCommandBuffers(cmd_pool, cmd_bufs);

        return *gpu_best_distance;
    }

  private:
    const GraphBuffers<GraphT> &graph_buffers;
    NearFarBuffers &nearfar_buffers;

    Statistics &statistics;

    ComputePipeline relax_pipeline;
    ComputePipeline compact_pipeline;
    ComputePipeline prepare_dispatch_pipeline;

    vk::Device device;
    uint32_t delta;
    uint32_t relax_batch_size;
    uint32_t workgroup_size;
    std::vector<vk::CommandBuffer> relax_cmd_bufs;
    std::vector<vk::CommandBuffer> compact_cmd_bufs;
};

} // namespace gpusssp::gpu

#endif
