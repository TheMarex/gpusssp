#ifndef GPUSSSP_GPU_DELTASETP_HPP
#define GPUSSSP_GPU_DELTASETP_HPP

#include <array>
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

using DeltaStepTracer = Tracer<DeltaStepPayload>;

template <typename GraphT> class DeltaStep
{
    static constexpr const size_t DEFAULT_WORKGROUP_SIZE = 64u;
    struct PushConsts
    {
        uint32_t src_node;
        uint32_t dst_node;
        uint32_t n;
        uint32_t bucket_idx;
        uint32_t delta;
        uint32_t max_weight;
    };

  public:
    DeltaStep(const GraphBuffers<GraphT> &graph_buffers,
              DeltaStepBuffers &deltastep_buffers,
              vk::Device &device,
              Statistics &statistics,
              uint32_t workgroup_size = DEFAULT_WORKGROUP_SIZE)
        : graph_buffers(graph_buffers), deltastep_buffers(deltastep_buffers),
          statistics(statistics), device(device), workgroup_size(workgroup_size)
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
        auto [dist_buffer,
              results_buffer,
              changed_buffer_0,
              changed_buffer_1,
              min_max_changed_id_0,
              min_max_changed_id_1,
              dispatch_buffer] = deltastep_buffers.buffers();
        auto statistics_buffer = statistics.buffer();

        main_pipeline = create_compute_pipeline<PushConsts>(device,
                                                            "delta_step.spv",
                                                            {{first_edges_buffer,
                                                              targets_buffer,
                                                              weights_buffer,
                                                              dist_buffer,
                                                              changed_buffer_0,
                                                              changed_buffer_1,
                                                              min_max_changed_id_0,
                                                              min_max_changed_id_1,
                                                              statistics_buffer},
                                                             {first_edges_buffer,
                                                              targets_buffer,
                                                              weights_buffer,
                                                              dist_buffer,
                                                              changed_buffer_1,
                                                              changed_buffer_0,
                                                              min_max_changed_id_1,
                                                              min_max_changed_id_0,
                                                              statistics_buffer}},
                                                            {workgroup_size});

        prepare_dispatch_pipeline = create_compute_pipeline(
            device,
            "deltastep_prepare_dispatch.spv",
            {{min_max_changed_id_1, dispatch_buffer}, {min_max_changed_id_0, dispatch_buffer}},
            {workgroup_size});
    }

    template <typename QueueT>
    uint32_t run(vk::CommandPool &cmd_pool,
                 QueueT &queue,
                 uint32_t src_node,
                 uint32_t dst_node,
                 uint32_t delta,
                 uint32_t batch_size = 64,
                 DeltaStepTracer *tracer = nullptr)
    {
        if (tracer)
        {
            tracer->start();
        }

        vk::CommandBuffer cmd_buf =
            device.allocateCommandBuffers({cmd_pool, vk::CommandBufferLevel::ePrimary, 1})[0];

        const std::size_t MAX_BUCKETS = common::INF_WEIGHT / delta - 1;
        const uint32_t min_max_init[] = {UINT32_MAX, 0};

        uint32_t *gpu_min_max_changed_id_0 = deltastep_buffers.min_max_changed_id_0();
        uint32_t *gpu_min_max_changed_id_1 = deltastep_buffers.min_max_changed_id_1();
        uint32_t *gpu_best_distance = deltastep_buffers.best_distance();
        uint32_t *gpu_max_distance = deltastep_buffers.max_distance();
        auto num_nodes = (uint32_t)graph_buffers.num_nodes();

        *gpu_best_distance = common::INF_WEIGHT;
        *gpu_max_distance = 0u;

        auto [dist_buffer,
              results_buffer,
              changed_buffer_0,
              changed_buffer_1,
              min_max_changed_id_buffer_0,
              min_max_changed_id_buffer_1,
              dispatch_buffer] = deltastep_buffers.buffers();

        const std::array<vk::BufferCopy, 2> results_copy = {
            vk::BufferCopy{dst_node * sizeof(uint32_t), 0, sizeof(uint32_t)},
            vk::BufferCopy{num_nodes * sizeof(uint32_t), sizeof(uint32_t), sizeof(uint32_t)}};

        auto record_start = common::Statistics::get().start(
            common::StatisticsEvent::DELTASTEP_CMDBUF_RECORD_DURATION);
        cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        cmd_buf.fillBuffer(dist_buffer, 0, num_nodes * sizeof(uint32_t), common::INF_WEIGHT);
        // initialize source wiht 0
        cmd_buf.fillBuffer(dist_buffer, src_node * sizeof(uint32_t), sizeof(uint32_t), 0);
        // initialize max_distance with 0
        cmd_buf.fillBuffer(dist_buffer, num_nodes * sizeof(uint32_t), sizeof(uint32_t), 0);
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

        for (uint32_t bucket = 0; bucket < MAX_BUCKETS; bucket++)
        {
            common::Statistics::get().count(common::StatisticsEvent::DELTASTEP_BUCKET);
            PushConsts pc{src_node, dst_node, num_nodes, bucket, delta, delta};

            auto previous_changed_buffer = changed_buffer_0;
            auto current_changed_buffer = changed_buffer_1;
            auto previous_min_max_changed_id_buffer = min_max_changed_id_buffer_0;
            auto current_min_max_changed_id_buffer = min_max_changed_id_buffer_1;
            auto *gpu_prev_min_changed_id = gpu_min_max_changed_id_0;
            auto *gpu_current_min_changed_id = gpu_min_max_changed_id_1;
            auto *gpu_prev_max_changed_id = gpu_min_max_changed_id_0 + 1;
            auto *gpu_current_max_changed_id = gpu_min_max_changed_id_1 + 1;
            auto prev_dispatch_desc_set = prepare_dispatch_pipeline.descriptor_sets[0];
            auto current_dispatch_desc_set = prepare_dispatch_pipeline.descriptor_sets[1];
            auto prev_desc_set = main_pipeline.descriptor_sets[0];
            auto current_desc_set = main_pipeline.descriptor_sets[1];

            auto record_start = common::Statistics::get().start(
                common::StatisticsEvent::DELTASTEP_CMDBUF_RECORD_DURATION);
            cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            // For each bucket we need to do the first pass with all nodes.
            // Essentially the first run identifies the nodes in the bucket,
            // All subsequent runs are limited to that.
            *gpu_prev_min_changed_id = 0;
            *gpu_prev_max_changed_id = num_nodes - 1;

            cmd_buf.fillBuffer(previous_changed_buffer, 0, VK_WHOLE_SIZE, UINT32_MAX);
            cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                    vk::PipelineStageFlagBits::eComputeShader,
                                    vk::DependencyFlags{},
                                    vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite,
                                                      vk::AccessFlagBits::eShaderRead |
                                                          vk::AccessFlagBits::eShaderWrite},
                                    {},
                                    {});

            do
            {

                for (uint32_t batch_iter = 0; batch_iter < batch_size; ++batch_iter)
                {

                    cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute,
                                         prepare_dispatch_pipeline.pipeline);
                    cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                               prepare_dispatch_pipeline.layout,
                                               0,
                                               current_dispatch_desc_set,
                                               {});
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

                    cmd_buf.fillBuffer(current_changed_buffer, 0, VK_WHOLE_SIZE, 0);
                    cmd_buf.updateBuffer(
                        current_min_max_changed_id_buffer, 0, sizeof(min_max_init), min_max_init);
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
                                            vk::PipelineStageFlagBits::eHost,
                                            vk::DependencyFlags{},
                                            vk::MemoryBarrier{vk::AccessFlagBits::eShaderWrite,
                                                              vk::AccessFlagBits::eHostRead},
                                            {},
                                            {});

                    std::swap(previous_changed_buffer, current_changed_buffer);
                    std::swap(previous_min_max_changed_id_buffer,
                              current_min_max_changed_id_buffer);
                    std::swap(gpu_prev_min_changed_id, gpu_current_min_changed_id);
                    std::swap(gpu_prev_max_changed_id, gpu_current_max_changed_id);
                    std::swap(prev_dispatch_desc_set, current_dispatch_desc_set);
                    std::swap(prev_desc_set, current_desc_set);
                }

                cmd_buf.end();
                common::Statistics::get().stop(
                    common::StatisticsEvent::DELTASTEP_CMDBUF_RECORD_DURATION, record_start);
                queue.submit(vk::SubmitInfo{0, nullptr, nullptr, 1, &cmd_buf});
                queue.waitIdle();

                if (tracer)
                {
                    // the prev buffer is the one we want to visualize since that is where the
                    // changed nodes are after the swap
                    tracer->signal_and_wait(
                        {bucket, gpu_prev_max_changed_id == gpu_min_max_changed_id_0 ? 0u : 1u});
                }

                common::log_debug() << bucket << " changed " << *gpu_prev_min_changed_id << "-"
                                    << *gpu_prev_max_changed_id << std::endl;

                // start a new command buffer either for next iteration here or the heavy pass
                record_start = common::Statistics::get().start(
                    common::StatisticsEvent::DELTASTEP_CMDBUF_RECORD_DURATION);
                cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

                // since we do this after the swap we need to look at prev not current
            } while (*gpu_prev_min_changed_id < num_nodes);

            common::Statistics::get().count(common::StatisticsEvent::DELTASTEP_HEAVY);
            cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, main_pipeline.pipeline);
            cmd_buf.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, main_pipeline.layout, 0, current_desc_set, {});

            *gpu_prev_min_changed_id = 0;
            *gpu_prev_max_changed_id = num_nodes - 1;
            cmd_buf.fillBuffer(previous_changed_buffer, 0, VK_WHOLE_SIZE, UINT32_MAX);
            cmd_buf.fillBuffer(current_changed_buffer, 0, VK_WHOLE_SIZE, 0);
            cmd_buf.updateBuffer(
                current_min_max_changed_id_buffer, 0, sizeof(min_max_init), min_max_init);
            cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                    vk::PipelineStageFlagBits::eComputeShader,
                                    vk::DependencyFlags{},
                                    vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite,
                                                      vk::AccessFlagBits::eShaderRead |
                                                          vk::AccessFlagBits::eShaderWrite},
                                    {},
                                    {});

            pc.max_weight = UINT32_MAX;
            cmd_buf.pushConstants(
                main_pipeline.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
            cmd_buf.dispatch((num_nodes + workgroup_size - 1) / workgroup_size, 1, 1);
            cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                    vk::PipelineStageFlagBits::eTransfer,
                                    vk::DependencyFlags{},
                                    vk::MemoryBarrier{vk::AccessFlagBits::eShaderWrite,
                                                      vk::AccessFlagBits::eTransferRead},
                                    {},
                                    {});
            cmd_buf.copyBuffer(dist_buffer, results_buffer, results_copy);
            cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer |
                                        vk::PipelineStageFlagBits::eComputeShader,
                                    vk::PipelineStageFlagBits::eHost,
                                    vk::DependencyFlags{},
                                    vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite |
                                                          vk::AccessFlagBits::eShaderWrite,
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
                // we didn't do a swap before this, we want to display the current buffer
                tracer->signal_and_wait(
                    {bucket, gpu_current_max_changed_id == gpu_min_max_changed_id_0 ? 0u : 1u});
            }

            common::log_debug() << bucket << " heavy changed " << *gpu_current_min_changed_id << "-"
                                << *gpu_current_max_changed_id << " max " << *gpu_max_distance
                                << " best " << *gpu_best_distance << std::endl;

            if (*gpu_best_distance != common::INF_WEIGHT)
            {
                // If the distance is smaller then the current bucket,
                // we have already settled the destination
                if (*gpu_best_distance < bucket * delta)
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
    uint32_t workgroup_size;
};

} // namespace gpusssp::gpu

#endif
