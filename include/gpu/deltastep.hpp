#ifndef GPUSSSP_GPU_DELTASETP_HPP
#define GPUSSSP_GPU_DELTASETP_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vulkan/vulkan.hpp>

#include "common/constants.hpp"
#include "common/logger.hpp"
#include "common/statistics.hpp"
#include "gpu/deltastep_buffers.hpp"
#include "gpu/graph_buffers.hpp"
#include "gpu/shader.hpp"
#include "gpu/statistics.hpp"
#include "gpu/tracer.hpp"

namespace gpusssp::gpu
{

struct DeltaStepPayload
{
    uint32_t bucket_index;
    // Since we use double-buffering we need to know which buffer is current
    uint32_t buffer_index;
};

using DeltaStepTracer = Tracer<DeltaStepPayload>; // NOLINT

template <typename GraphT> class DeltaStep
{
    static constexpr const size_t DEFAULT_WORKGROUP_SIZE = 64u;
    static constexpr const uint32_t DEFAULT_RELAX_BATCH_SIZE = 64u;
    struct PushConsts
    {
        uint32_t src_node;
        uint32_t dst_node;
        uint32_t n;
        uint32_t bucket_idx;
        uint32_t delta;
    };

  public:
    DeltaStep(const GraphBuffers<GraphT> &graph_buffers,
              DeltaStepBuffers &deltastep_buffers,
              vk::Device &device,
              Statistics &statistics,
              uint32_t delta,
              uint32_t relax_batch_size = DEFAULT_RELAX_BATCH_SIZE,
              uint32_t workgroup_size = DEFAULT_WORKGROUP_SIZE)
        : graph_buffers(graph_buffers), deltastep_buffers(deltastep_buffers),
          statistics(statistics), device(device), delta(delta),
          relax_batch_size(relax_batch_size), workgroup_size(workgroup_size)
    {
    }

    ~DeltaStep()
    {
        device.destroyShaderModule(main_pipeline.shader);
        device.destroyPipeline(main_pipeline.pipeline);
        device.destroyPipelineLayout(main_pipeline.layout);
        device.destroyDescriptorSetLayout(main_pipeline.descriptor_set_layout);
        device.destroyDescriptorPool(main_pipeline.descriptor_pool);

        device.destroyShaderModule(prepare_dispatch_pipeline.shader);
        device.destroyPipeline(prepare_dispatch_pipeline.pipeline);
        device.destroyPipelineLayout(prepare_dispatch_pipeline.layout);
        device.destroyDescriptorSetLayout(prepare_dispatch_pipeline.descriptor_set_layout);
        device.destroyDescriptorPool(prepare_dispatch_pipeline.descriptor_pool);
    }

    void initialize()
    {
        auto [first_edges_buffer, targets_buffer, weights_buffer] = graph_buffers.buffers();
        auto dist_buffer = deltastep_buffers.dist_buffer();
        auto [changed_buffer_0, changed_buffer_1] = deltastep_buffers.changed_buffers();
        auto dispatch_buffer = deltastep_buffers.dispatch_buffer();
        auto statistics_buffer = statistics.buffer();

        main_pipeline = create_compute_pipeline<PushConsts>(device,
                                                            "delta_step.spv",
                                                            {{first_edges_buffer,
                                                              targets_buffer,
                                                              weights_buffer,
                                                              dist_buffer,
                                                              changed_buffer_0,
                                                              changed_buffer_1,
                                                              statistics_buffer},
                                                             {first_edges_buffer,
                                                              targets_buffer,
                                                              weights_buffer,
                                                              dist_buffer,
                                                              changed_buffer_1,
                                                              changed_buffer_0,
                                                              statistics_buffer}},
                                                            {workgroup_size});

        prepare_dispatch_pipeline = create_compute_pipeline<uint32_t>(
            device,
            "deltastep_prepare_dispatch.spv",
            {{changed_buffer_1, dispatch_buffer}, {changed_buffer_0, dispatch_buffer}},
            {workgroup_size});
    }

    template <typename QueueT>
    uint32_t run(vk::CommandPool &cmd_pool,
                 QueueT &queue,
                 uint32_t src_node,
                 uint32_t dst_node,
                 DeltaStepTracer *tracer = nullptr)
    {
        if (tracer)
        {
            tracer->start();
        }

        vk::CommandBuffer cmd_buf =
            device.allocateCommandBuffers({cmd_pool, vk::CommandBufferLevel::ePrimary, 1})[0];

        uint32_t num_nodes = graph_buffers.num_nodes();
        const std::size_t max_buckets = common::INF_WEIGHT / delta;

        uint32_t *gpu_min_changed_id = deltastep_buffers.min_changed_id();
        uint32_t *gpu_max_changed_id = deltastep_buffers.max_changed_id();
        uint32_t *gpu_best_distance = deltastep_buffers.best_distance();
        uint32_t *gpu_max_distance = deltastep_buffers.max_distance();

        auto dispatch_buffer = deltastep_buffers.dispatch_buffer();

        auto record_start =
            common::Statistics::start(common::StatisticsEvent::DELTASTEP_CMDBUF_RECORD_DURATION);
        cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        deltastep_buffers.cmd_init_dist(cmd_buf, src_node);
        cmd_buf.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
            vk::DependencyFlags{},
            vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite,
                              vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                                  vk::AccessFlagBits::eTransferRead},
            {},
            {});

        cmd_buf.end();
        common::Statistics::get().stop(common::StatisticsEvent::DELTASTEP_CMDBUF_RECORD_DURATION,
                                       record_start);
        queue.submit(vk::SubmitInfo{0, nullptr, nullptr, 1, &cmd_buf});
        queue.waitIdle();

        for (uint32_t bucket = 0; bucket < max_buckets; bucket++)
        {
            common::Statistics::get().count(common::StatisticsEvent::DELTASTEP_BUCKET);
            PushConsts pc{src_node, dst_node, num_nodes, bucket, delta};

            uint32_t prev_buffer_idx = 0;
            uint32_t curr_buffer_idx = 1;
            auto prev_dispatch_desc_set = prepare_dispatch_pipeline.descriptor_sets[0];
            auto current_dispatch_desc_set = prepare_dispatch_pipeline.descriptor_sets[1];
            auto prev_desc_set = main_pipeline.descriptor_sets[0];
            auto current_desc_set = main_pipeline.descriptor_sets[1];

            auto record_start = common::Statistics::start(
                common::StatisticsEvent::DELTASTEP_CMDBUF_RECORD_DURATION);
            cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            // For each bucket we need to do the first pass with all nodes.
            // Essentially the first run identifies the nodes in the bucket,
            // All subsequent runs are limited to that.
            *gpu_min_changed_id = 0;
            *gpu_max_changed_id = num_nodes - 1;
            *gpu_best_distance = common::INF_WEIGHT;

            deltastep_buffers.cmd_init_changed(cmd_buf, prev_buffer_idx);

            cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                    vk::PipelineStageFlagBits::eComputeShader,
                                    vk::DependencyFlags{},
                                    vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite,
                                                      vk::AccessFlagBits::eShaderRead |
                                                          vk::AccessFlagBits::eShaderWrite},
                                    {},
                                    {});

            bool converged = false;
            while (!converged)
            {

                for (uint32_t batch_iter = 0; batch_iter < relax_batch_size; ++batch_iter)
                {

                    cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute,
                                         prepare_dispatch_pipeline.pipeline);
                    cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                               prepare_dispatch_pipeline.layout,
                                               0,
                                               current_dispatch_desc_set,
                                               {});
                    cmd_buf.pushConstants(prepare_dispatch_pipeline.layout,
                                          vk::ShaderStageFlagBits::eCompute,
                                          0,
                                          4,
                                          &num_nodes);
                    cmd_buf.dispatch(1, 1, 1);
                    cmd_buf.pipelineBarrier(
                        vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader,
                        vk::DependencyFlags{},
                        vk::MemoryBarrier{vk::AccessFlagBits::eShaderWrite,
                                          vk::AccessFlagBits::eShaderRead |
                                              vk::AccessFlagBits::eIndirectCommandRead},
                        {},
                        {});

                    cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, main_pipeline.pipeline);
                    cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                               main_pipeline.layout,
                                               0,
                                               current_desc_set,
                                               {});

                    deltastep_buffers.cmd_clear_changed(cmd_buf, curr_buffer_idx);
                    cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                            vk::PipelineStageFlagBits::eComputeShader,
                                            vk::DependencyFlags{},
                                            vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite,
                                                              vk::AccessFlagBits::eShaderRead |
                                                                  vk::AccessFlagBits::eShaderWrite},
                                            {},
                                            {});

                    cmd_buf.pushConstants(main_pipeline.layout,
                                          vk::ShaderStageFlagBits::eCompute,
                                          0,
                                          sizeof(pc),
                                          &pc);
                    cmd_buf.dispatchIndirect(dispatch_buffer, 0);
                    cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                            vk::PipelineStageFlagBits::eComputeShader,
                                            vk::DependencyFlags{},
                                            vk::MemoryBarrier{vk::AccessFlagBits::eShaderWrite,
                                                              vk::AccessFlagBits::eShaderRead},
                                            {},
                                            {});

                    std::swap(prev_buffer_idx, curr_buffer_idx);
                    std::swap(prev_dispatch_desc_set, current_dispatch_desc_set);
                    std::swap(prev_desc_set, current_desc_set);
                }

                cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                        vk::PipelineStageFlagBits::eTransfer,
                                        vk::DependencyFlags{},
                                        vk::MemoryBarrier{vk::AccessFlagBits::eShaderWrite,
                                                          vk::AccessFlagBits::eTransferRead},
                                        {},
                                        {});

                // After the swap, prev_buffer_idx has the latest results
                deltastep_buffers.cmd_sync_results(cmd_buf, dst_node, prev_buffer_idx);

                cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                        vk::PipelineStageFlagBits::eHost,
                                        vk::DependencyFlags{},
                                        vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite,
                                                          vk::AccessFlagBits::eHostRead},
                                        {},
                                        {});

                cmd_buf.end();
                common::Statistics::get().stop(
                    common::StatisticsEvent::DELTASTEP_CMDBUF_RECORD_DURATION, record_start);
                queue.submit(vk::SubmitInfo{0, nullptr, nullptr, 1, &cmd_buf});
                queue.waitIdle();

                if (tracer)
                {
                    tracer->signal_and_wait(
                        {.bucket_index = bucket, .buffer_index = prev_buffer_idx});
                }

                uint32_t range_size = std::max(0L,
                                               static_cast<int64_t>(*gpu_max_changed_id) -
                                                   static_cast<int64_t>(*gpu_min_changed_id));
                common::Statistics::get().sum(common::StatisticsEvent::DELTASTEP_RANGE, range_size);

                common::log_debug()
                    << bucket << " changed " << *gpu_min_changed_id << "-" << *gpu_max_changed_id
                    << " ~ " << static_cast<double>(range_size) / num_nodes << '\n';

                converged = *gpu_min_changed_id >= num_nodes;
                if (!converged)
                {
                    record_start = common::Statistics::start(
                        common::StatisticsEvent::DELTASTEP_CMDBUF_RECORD_DURATION);
                    cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
                }
            }

            if (*gpu_best_distance != common::INF_WEIGHT)
            {
                // If the distance is smaller then the current bucket,
                // we have already settled the destination
                if (*gpu_best_distance < (bucket + 1) * delta)
                {
                    break;
                }
            }

            // If the maximum node distance is lower than the next bucket all other buckets
            // will be empty -> dst_node is unreachable.
            if (*gpu_max_distance < (bucket + 1) * delta)
            {
                break;
            }
        }

        if (tracer)
        {
            tracer->finish();
        }

        return *gpu_best_distance;
    }

  private:
    const GraphBuffers<GraphT> &graph_buffers;
    DeltaStepBuffers &deltastep_buffers;
    Statistics &statistics;

    ComputePipeline main_pipeline;
    ComputePipeline prepare_dispatch_pipeline;

    vk::Device &device;
    uint32_t delta;
    uint32_t relax_batch_size;
    uint32_t workgroup_size;
};

} // namespace gpusssp::gpu

#endif
