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
              dispatch_relax_buffer,
              processed_buffer] = nearfar_buffers.buffers();
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
                                                               statistics_buffer},
                                                              {first_edges_buffer,
                                                               targets_buffer,
                                                               weights_buffer,
                                                               dist_buffer,
                                                               near_1_buffer,
                                                               near_0_buffer,
                                                               far_0_buffer,
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
                                                                    statistics_buffer},
                                                               },
                                                               {workgroup_size});

        prepare_dispatch_pipeline = create_compute_pipeline<uint32_t>(
            device,
            "nearfar_prepare_dispatch.spv",
            {{near_0_buffer, dispatch_relax_buffer}, {near_1_buffer, dispatch_relax_buffer}},
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
        auto init_start = common::Statistics::start(common::StatisticsEvent::NEARFAR_INIT_DURATION);

        vk::CommandBuffer cmd_buf =
            device.allocateCommandBuffers({cmd_pool, vk::CommandBufferLevel::ePrimary, 1})[0];

        uint32_t *gpu_best_distance = nearfar_buffers.best_distance();
        uint32_t *gpu_num_near = nearfar_buffers.num_near();
        uint32_t *gpu_num_far = nearfar_buffers.num_far();
        auto num_nodes = static_cast<uint32_t>(graph_buffers.num_nodes());

        auto [dist_buffer,
              results_buffer,
              near_0_buffer,
              near_1_buffer,
              far_0_buffer,
              far_1_buffer,
              dispatch_relax_buffer,
              processed_buffer] = nearfar_buffers.buffers();

        vk::BufferCopy results_copy{dst_node * sizeof(uint32_t), 0, sizeof(uint32_t)};
        vk::BufferCopy near_count_copy{
            num_nodes * sizeof(uint32_t), 1 * sizeof(uint32_t), sizeof(uint32_t)};
        vk::BufferCopy far_count_copy{
            num_nodes * sizeof(uint32_t), 2 * sizeof(uint32_t), sizeof(uint32_t)};

        auto record_0_start =
            common::Statistics::start(common::StatisticsEvent::NEARFAR_CMDBUF_RECORD_DURATION);
        cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        cmd_buf.fillBuffer(dist_buffer, 0, num_nodes * sizeof(uint32_t), common::INF_WEIGHT);
        cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                vk::PipelineStageFlagBits::eTransfer,
                                vk::DependencyFlags{},
                                vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite,
                                                  vk::AccessFlagBits::eTransferWrite},
                                {},
                                {});

        uint32_t zero = 0;
        uint32_t one = 1;
        cmd_buf.updateBuffer(dist_buffer, src_node * sizeof(uint32_t), sizeof(uint32_t), &zero);

        cmd_buf.updateBuffer(near_0_buffer, 0, sizeof(uint32_t), &src_node);
        // initialize near_0 counter with 1
        cmd_buf.updateBuffer(near_0_buffer, num_nodes * sizeof(uint32_t), sizeof(uint32_t), &one);
        // initialize far_0 counter with 0
        cmd_buf.updateBuffer(far_0_buffer, num_nodes * sizeof(uint32_t), sizeof(uint32_t), &zero);

        cmd_buf.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
            vk::DependencyFlags{},
            vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite,
                              vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                                  vk::AccessFlagBits::eTransferRead},
            {},
            {});

        cmd_buf.copyBuffer(dist_buffer, results_buffer, 1, &results_copy);
        cmd_buf.copyBuffer(near_0_buffer, results_buffer, 1, &near_count_copy);
        cmd_buf.copyBuffer(far_0_buffer, results_buffer, 1, &far_count_copy);

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
        queue.submit(vk::SubmitInfo{0, nullptr, nullptr, 1, &cmd_buf});
        queue.waitIdle();

        uint32_t current_near_buffer_idx = 0;
        uint32_t phase = 0;

        common::Statistics::get().stop(common::StatisticsEvent::NEARFAR_INIT_DURATION, init_start);

        while (true)
        {
            common::Statistics::get().count(common::StatisticsEvent::NEARFAR_PHASE);

            auto relax_start =
                common::Statistics::start(common::StatisticsEvent::NEARFAR_RELAX_DURATION);

            while (*gpu_num_near > 0)
            {
                common::Statistics::get().count(common::StatisticsEvent::NEARFAR_RELAX);
                common::log_debug() << phase << " " << *gpu_num_near << " best distance "
                                    << *gpu_best_distance << '\n';
                auto record_1_start = common::Statistics::start(
                    common::StatisticsEvent::NEARFAR_CMDBUF_RECORD_DURATION);
                cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

                for (uint32_t batch_iter = 0; batch_iter < relax_batch_size; ++batch_iter)
                {
                    auto relax_desc_set = relax_pipeline.descriptor_sets[current_near_buffer_idx];

                    auto next_near_buffer =
                        current_near_buffer_idx == 0 ? near_1_buffer : near_0_buffer;

                    // clear size of next_near
                    cmd_buf.fillBuffer(
                        next_near_buffer, num_nodes * sizeof(uint32_t), sizeof(uint32_t), 0);
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

                    cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                            vk::PipelineStageFlagBits::eTransfer,
                                            vk::DependencyFlags{},
                                            vk::MemoryBarrier{vk::AccessFlagBits::eShaderWrite,
                                                              vk::AccessFlagBits::eTransferRead},
                                            {},
                                            {});

                    cmd_buf.copyBuffer(dist_buffer, results_buffer, 1, &results_copy);
                    cmd_buf.copyBuffer(next_near_buffer, results_buffer, 1, &near_count_copy);
                    cmd_buf.copyBuffer(far_0_buffer, results_buffer, 1, &far_count_copy);

                    cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                            vk::PipelineStageFlagBits::eHost |
                                                vk::PipelineStageFlagBits::eComputeShader,
                                            vk::DependencyFlags{},
                                            vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite,
                                                              vk::AccessFlagBits::eHostRead |
                                                                  vk::AccessFlagBits::eShaderRead},
                                            {},
                                            {});

                    current_near_buffer_idx = 1 - current_near_buffer_idx;
                }

                common::Statistics::get().stop(
                    common::StatisticsEvent::NEARFAR_CMDBUF_RECORD_DURATION, record_1_start);

                cmd_buf.end();
                queue.submit(vk::SubmitInfo{0, nullptr, nullptr, 1, &cmd_buf});
                queue.waitIdle();
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
                common::Statistics::start(common::StatisticsEvent::NEARFAR_COMPACT_DURATION);

            if (*gpu_num_far == 0)
            {
                break;
            }

            common::log_debug() << phase << " far " << *gpu_num_far << " best distance "
                                << *gpu_best_distance << '\n';

            auto record_2_start =
                common::Statistics::start(common::StatisticsEvent::NEARFAR_CMDBUF_RECORD_DURATION);
            cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

            auto compact_desc_set = compact_pipeline.descriptor_sets[0];

            cmd_buf.fillBuffer(near_0_buffer, num_nodes * sizeof(uint32_t), sizeof(uint32_t), 0);
            cmd_buf.fillBuffer(far_1_buffer, num_nodes * sizeof(uint32_t), sizeof(uint32_t), 0);
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
            cmd_buf.dispatch((*gpu_num_far + workgroup_size - 1) / workgroup_size, 1, 1);

            cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                    vk::PipelineStageFlagBits::eTransfer,
                                    vk::DependencyFlags{},
                                    vk::MemoryBarrier{vk::AccessFlagBits::eShaderWrite,
                                                      vk::AccessFlagBits::eTransferRead},
                                    {},
                                    {});

            cmd_buf.copyBuffer(dist_buffer, results_buffer, 1, &results_copy);
            cmd_buf.copyBuffer(near_0_buffer, results_buffer, 1, &near_count_copy);
            cmd_buf.copyBuffer(far_1_buffer, results_buffer, 1, &far_count_copy);

            cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                    vk::PipelineStageFlagBits::eHost,
                                    vk::DependencyFlags{},
                                    vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite,
                                                      vk::AccessFlagBits::eHostRead},
                                    {},
                                    {});

            cmd_buf.end();
            common::Statistics::get().stop(common::StatisticsEvent::NEARFAR_CMDBUF_RECORD_DURATION,
                                           record_2_start);

            queue.submit(vk::SubmitInfo{0, nullptr, nullptr, 1, &cmd_buf});
            queue.waitIdle();

            current_near_buffer_idx = 0;

            common::log_debug() << phase << " compacted to: far " << *gpu_num_far << " near "
                                << *gpu_num_near << '\n';

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
