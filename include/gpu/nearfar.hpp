#ifndef GPUSSSP_GPU_NEARFAR_HPP
#define GPUSSSP_GPU_NEARFAR_HPP

#include <vulkan/vulkan.hpp>

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
    struct PushConsts
    {
        uint32_t src_node;
        uint32_t dst_node;
        uint32_t n;
        uint32_t phase;
        uint32_t delta;
    };

  public:
    NearFar(const GraphBuffers<GraphT> &graph_buffers,
            NearFarBuffers &nearfar_buffers,
            vk::Device &device,
            Statistics &statistics,
            uint32_t workgroup_size = DEFAULT_WORKGROUP_SIZE)
        : graph_buffers(graph_buffers), nearfar_buffers(nearfar_buffers), statistics(statistics),
          device(device), workgroup_size(workgroup_size)
    {
    }

    ~NearFar()
    {
        device.destroyShaderModule(relax_pipeline.shader);
        device.destroyPipeline(relax_pipeline.pipeline);
        device.destroyPipelineLayout(relax_pipeline.layout);
        device.destroyDescriptorSetLayout(relax_pipeline.descriptor_set_layout);
        device.destroyDescriptorPool(relax_pipeline.descriptor_pool);

        device.destroyShaderModule(compact_pipeline.shader);
        device.destroyPipeline(compact_pipeline.pipeline);
        device.destroyPipelineLayout(compact_pipeline.layout);
        device.destroyDescriptorSetLayout(compact_pipeline.descriptor_set_layout);
        device.destroyDescriptorPool(compact_pipeline.descriptor_pool);

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
              near_0_buffer,
              near_1_buffer,
              far_0_buffer,
              far_1_buffer,
              counters_buffer,
              dispatch_relax_buffer,
              processed_buffer] = nearfar_buffers.buffers();
        auto statistics_buffer = statistics.buffer();

        relax_pipeline = create_compute_pipeline<PushConsts>(device,
                                                             "nearfar_relax.spv",
                                                             {{first_edges_buffer,
                                                               targets_buffer,
                                                               weights_buffer,
                                                               dist_buffer,
                                                               results_buffer,
                                                               near_0_buffer,
                                                               near_1_buffer,
                                                               far_0_buffer,
                                                               counters_buffer,
                                                               statistics_buffer},
                                                              {first_edges_buffer,
                                                               targets_buffer,
                                                               weights_buffer,
                                                               dist_buffer,
                                                               results_buffer,
                                                               near_0_buffer,
                                                               near_1_buffer,
                                                               far_1_buffer,
                                                               counters_buffer,
                                                               statistics_buffer},
                                                              {first_edges_buffer,
                                                               targets_buffer,
                                                               weights_buffer,
                                                               dist_buffer,
                                                               results_buffer,
                                                               near_1_buffer,
                                                               near_0_buffer,
                                                               far_0_buffer,
                                                               counters_buffer,
                                                               statistics_buffer},
                                                              {first_edges_buffer,
                                                               targets_buffer,
                                                               weights_buffer,
                                                               dist_buffer,
                                                               results_buffer,
                                                               near_1_buffer,
                                                               near_0_buffer,
                                                               far_1_buffer,
                                                               counters_buffer,
                                                               statistics_buffer}},
                                                             {workgroup_size});

        compact_pipeline = create_compute_pipeline<PushConsts>(device,
                                                               "nearfar_compact.spv",
                                                               {{dist_buffer,
                                                                 results_buffer,
                                                                 far_0_buffer,
                                                                 near_0_buffer,
                                                                 far_1_buffer,
                                                                 counters_buffer,
                                                                 processed_buffer,
                                                                 statistics_buffer},
                                                                {dist_buffer,
                                                                 results_buffer,
                                                                 far_1_buffer,
                                                                 near_0_buffer,
                                                                 far_0_buffer,
                                                                 counters_buffer,
                                                                 processed_buffer,
                                                                 statistics_buffer}},
                                                               {workgroup_size});

        prepare_dispatch_pipeline =
            create_compute_pipeline(device,
                                    "nearfar_prepare_dispatch.spv",
                                    {{counters_buffer, dispatch_relax_buffer}},
                                    {workgroup_size});
    }

    template <typename QueueT>
    uint32_t run(vk::CommandPool &cmd_pool,
                 QueueT &queue,
                 uint32_t src_node,
                 uint32_t dst_node,
                 uint32_t delta,
                 uint32_t relax_batch_size = 64)
    {
        auto init_start =
            common::Statistics::get().start(common::StatisticsEvent::NEARFAR_INIT_DURATION);

        vk::CommandBuffer cmd_buf =
            device.allocateCommandBuffers({cmd_pool, vk::CommandBufferLevel::ePrimary, 1})[0];

        uint32_t *gpu_best_distance = nearfar_buffers.best_distance();
        uint32_t *gpu_num_near = nearfar_buffers.num_near();
        uint32_t *gpu_num_far = nearfar_buffers.num_far();
        auto num_nodes = (uint32_t)graph_buffers.num_nodes();

        auto [dist_buffer,
              results_buffer,
              near_0_buffer,
              near_1_buffer,
              far_0_buffer,
              far_1_buffer,
              counters_buffer,
              dispatch_relax_buffer,
              processed_buffer] = nearfar_buffers.buffers();

        auto record_0_start = common::Statistics::get().start(
            common::StatisticsEvent::NEARFAR_CMDBUF_RECORD_DURATION);
        cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        if (src_node > 0)
        {
            cmd_buf.fillBuffer(dist_buffer, 0, src_node * sizeof(uint32_t), common::INF_WEIGHT);
        }
        cmd_buf.fillBuffer(dist_buffer, src_node * sizeof(uint32_t), sizeof(uint32_t), 0);
        if (src_node < num_nodes - 1)
        {
            cmd_buf.fillBuffer(
                dist_buffer, (src_node + 1) * sizeof(uint32_t), VK_WHOLE_SIZE, common::INF_WEIGHT);
        }

        cmd_buf.fillBuffer(near_0_buffer, 0, sizeof(uint32_t), src_node);

        cmd_buf.fillBuffer(counters_buffer, 0, sizeof(uint32_t), 1);
        cmd_buf.fillBuffer(counters_buffer, sizeof(uint32_t), 3 * sizeof(uint32_t), 0);

        cmd_buf.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlags{},
            vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite,
                              vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite},
            {},
            {});

        cmd_buf.end();
        common::Statistics::get().stop(common::StatisticsEvent::NEARFAR_CMDBUF_RECORD_DURATION,
                                       record_0_start);
        queue.submit(vk::SubmitInfo{0, nullptr, nullptr, 1, &cmd_buf});
        queue.waitIdle();

        *gpu_best_distance = common::INF_WEIGHT;

        uint32_t current_near_buffer = 0;
        uint32_t current_far_buffer = 0;
        uint32_t phase = 0;
        uint32_t num_near = 1;
        uint32_t num_far = 0;

        common::Statistics::get().stop(common::StatisticsEvent::NEARFAR_INIT_DURATION, init_start);

        while (true)
        {
            common::Statistics::get().count(common::StatisticsEvent::NEARFAR_PHASE);

            auto relax_start =
                common::Statistics::get().start(common::StatisticsEvent::NEARFAR_RELAX_DURATION);

            while (num_near > 0)
            {
                common::Statistics::get().count(common::StatisticsEvent::NEARFAR_RELAX);
                common::log_debug() << phase << " " << num_near << " best distance "
                                    << *gpu_best_distance << std::endl;
                auto record_1_start = common::Statistics::get().start(
                    common::StatisticsEvent::NEARFAR_CMDBUF_RECORD_DURATION);
                cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

                for (uint32_t batch_iter = 0; batch_iter < relax_batch_size; ++batch_iter)
                {
                    uint32_t relax_desc_idx = current_near_buffer * 2 + current_far_buffer;
                    auto relax_desc_set = relax_pipeline.descriptor_sets[relax_desc_idx];

                    // clear size of next_near
                    cmd_buf.fillBuffer(counters_buffer, 1 * sizeof(uint32_t), sizeof(uint32_t), 0);
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
                    cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                               prepare_dispatch_pipeline.layout,
                                               0,
                                               prepare_dispatch_pipeline.descriptor_sets[0],
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

                    cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, relax_pipeline.pipeline);
                    cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                               relax_pipeline.layout,
                                               0,
                                               relax_desc_set,
                                               {});

                    PushConsts pc{src_node, dst_node, num_nodes, phase, delta};
                    cmd_buf.pushConstants(relax_pipeline.layout,
                                          vk::ShaderStageFlagBits::eCompute,
                                          0,
                                          sizeof(pc),
                                          &pc);
                    cmd_buf.dispatchIndirect(dispatch_relax_buffer, 0);

                    cmd_buf.pipelineBarrier(
                        vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eTransfer,
                        vk::DependencyFlags{},
                        vk::MemoryBarrier{vk::AccessFlagBits::eShaderWrite,
                                          vk::AccessFlagBits::eTransferRead |
                                              vk::AccessFlagBits::eTransferWrite},
                        {},
                        {});

                    // copy size of next_near to current
                    vk::BufferCopy copy_region{
                        1 * sizeof(uint32_t), 0 * sizeof(uint32_t), sizeof(uint32_t)};
                    cmd_buf.copyBuffer(counters_buffer, counters_buffer, 1, &copy_region);

                    cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                            vk::PipelineStageFlagBits::eComputeShader,
                                            vk::DependencyFlags{},
                                            vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite,
                                                              vk::AccessFlagBits::eShaderRead |
                                                                  vk::AccessFlagBits::eShaderWrite},
                                            {},
                                            {});

                    current_near_buffer = 1 - current_near_buffer;
                }

                common::Statistics::get().stop(
                    common::StatisticsEvent::NEARFAR_CMDBUF_RECORD_DURATION, record_1_start);

                cmd_buf.end();
                queue.submit(vk::SubmitInfo{0, nullptr, nullptr, 1, &cmd_buf});
                queue.waitIdle();

                num_near = *gpu_num_near;
            }

            common::Statistics::get().stop(common::StatisticsEvent::NEARFAR_RELAX_DURATION,
                                           relax_start);

            if (*gpu_best_distance != common::INF_WEIGHT)
            {
                if (*gpu_best_distance < phase * delta)
                {
                    break;
                }
            }

            phase++;

            auto compact_start =
                common::Statistics::get().start(common::StatisticsEvent::NEARFAR_COMPACT_DURATION);

            num_far = *gpu_num_far;
            if (num_far == 0)
            {
                break;
            }

            common::log_debug() << phase << " far " << num_far << " best distance "
                                << *gpu_best_distance << std::endl;

            auto record_2_start = common::Statistics::get().start(
                common::StatisticsEvent::NEARFAR_CMDBUF_RECORD_DURATION);
            cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

            auto compact_desc_set = compact_pipeline.descriptor_sets[current_far_buffer];

            cmd_buf.fillBuffer(counters_buffer, 0 * sizeof(uint32_t), sizeof(uint32_t), 0);
            cmd_buf.fillBuffer(counters_buffer, 3 * sizeof(uint32_t), sizeof(uint32_t), 0);
            cmd_buf.fillBuffer(processed_buffer, 0, VK_WHOLE_SIZE, 0);
            cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                    vk::PipelineStageFlagBits::eComputeShader,
                                    vk::DependencyFlags{},
                                    vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite,
                                                      vk::AccessFlagBits::eShaderRead |
                                                          vk::AccessFlagBits::eShaderWrite},
                                    {},
                                    {});

            cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, compact_pipeline.pipeline);
            cmd_buf.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, compact_pipeline.layout, 0, compact_desc_set, {});

            PushConsts pc{src_node, dst_node, num_nodes, phase, delta};
            cmd_buf.pushConstants(
                compact_pipeline.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
            cmd_buf.dispatch((num_far + workgroup_size - 1) / workgroup_size, 1, 1);

            cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                    vk::PipelineStageFlagBits::eTransfer,
                                    vk::DependencyFlags{},
                                    vk::MemoryBarrier{vk::AccessFlagBits::eShaderWrite,
                                                      vk::AccessFlagBits::eTransferRead |
                                                          vk::AccessFlagBits::eTransferWrite},
                                    {},
                                    {});

            vk::BufferCopy copy_region{
                3 * sizeof(uint32_t), 2 * sizeof(uint32_t), sizeof(uint32_t)};
            cmd_buf.copyBuffer(counters_buffer, counters_buffer, 1, &copy_region);

            cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                    vk::PipelineStageFlagBits::eComputeShader,
                                    vk::DependencyFlags{},
                                    vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite,
                                                      vk::AccessFlagBits::eShaderRead |
                                                          vk::AccessFlagBits::eShaderWrite},
                                    {},
                                    {});

            cmd_buf.end();
            common::Statistics::get().stop(common::StatisticsEvent::NEARFAR_CMDBUF_RECORD_DURATION,
                                           record_2_start);

            queue.submit(vk::SubmitInfo{0, nullptr, nullptr, 1, &cmd_buf});
            queue.waitIdle();

            current_far_buffer = 1 - current_far_buffer;
            current_near_buffer = 0;

            num_near = *gpu_num_near;
            num_far = *gpu_num_far;

            common::log_debug() << phase << " compacted to: far " << num_far << " near " << num_near
                                << std::endl;

            common::Statistics::get().stop(common::StatisticsEvent::NEARFAR_COMPACT_DURATION,
                                           compact_start);
        }

        return *gpu_best_distance;
    }

  private:
    const GraphBuffers<GraphT> &graph_buffers;
    NearFarBuffers &nearfar_buffers;

    Statistics &statistics;

    ComputePipeline relax_pipeline;
    ComputePipeline compact_pipeline;
    ComputePipeline prepare_dispatch_pipeline;

    vk::Device &device;
    uint32_t workgroup_size;
};

} // namespace gpusssp::gpu

#endif
