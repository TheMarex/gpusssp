#ifndef GPUSSSP_GPU_NEARFAR_HPP
#define GPUSSSP_GPU_NEARFAR_HPP

#include <vulkan/vulkan.hpp>

#include "common/constants.hpp"
#include "common/shader.hpp"
#include "common/statistics.hpp"
#include "gpu/graph_buffers.hpp"
#include "gpu/memory.hpp"
#include "gpu/nearfar_buffers.hpp"
#include "gpu/statistics.hpp"

#include <iostream>

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
        device.destroyShaderModule(relax_shader);
        device.destroyPipeline(relax_pipeline);
        device.destroyPipelineLayout(relax_pipeline_layout);
        device.destroyDescriptorSetLayout(relax_desc_bundle.layout);
        device.destroyDescriptorPool(relax_desc_bundle.pool);

        device.destroyShaderModule(compact_shader);
        device.destroyPipeline(compact_pipeline);
        device.destroyPipelineLayout(compact_pipeline_layout);
        device.destroyDescriptorSetLayout(compact_desc_bundle.layout);
        device.destroyDescriptorPool(compact_desc_bundle.pool);

        device.destroyShaderModule(prepare_dispatch_shader);
        device.destroyPipeline(prepare_dispatch_pipeline);
        device.destroyPipelineLayout(prepare_dispatch_pipeline_layout);
        device.destroyDescriptorSetLayout(prepare_dispatch_desc_bundle.layout);
        device.destroyDescriptorPool(prepare_dispatch_desc_bundle.pool);
    }

    void initialize_relax_descriptor_sets()
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

        relax_desc_bundle = create_descriptor_sets(
            device,
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
              statistics_buffer}});
    }

    void initialize_compact_descriptor_sets()
    {
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

        compact_desc_bundle =
            create_descriptor_sets(device,
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
                                     statistics_buffer}});
    }

    void initialize_prepare_dispatch_descriptor_sets()
    {
        auto [dist_buffer,
              results_buffer,
              near_0_buffer,
              near_1_buffer,
              far_0_buffer,
              far_1_buffer,
              counters_buffer,
              dispatch_relax_buffer,
              processed_buffer] = nearfar_buffers.buffers();

        prepare_dispatch_desc_bundle =
            create_descriptor_sets(device, {{counters_buffer, dispatch_relax_buffer}});
    }

    void initialize()
    {
        initialize_relax_descriptor_sets();
        initialize_compact_descriptor_sets();
        initialize_prepare_dispatch_descriptor_sets();

        std::vector<uint32_t> relax_spv = common::read_spv("nearfar_relax.spv");
        relax_shader = device.createShaderModule({{}, relax_spv.size() * 4, relax_spv.data()});

        vk::PushConstantRange pcRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConsts)};
        relax_pipeline_layout =
            device.createPipelineLayout({{}, 1, &relax_desc_bundle.layout, 1, &pcRange});

        vk::SpecializationMapEntry spec_entry{0, 0, sizeof(uint32_t)};
        vk::SpecializationInfo spec_info{1, &spec_entry, sizeof(uint32_t), &workgroup_size};

        vk::PipelineShaderStageCreateInfo relax_shader_stage{
            {}, vk::ShaderStageFlagBits::eCompute, relax_shader, "main", &spec_info};

        relax_pipeline =
            device.createComputePipeline({}, {{}, relax_shader_stage, relax_pipeline_layout}).value;

        std::vector<uint32_t> compact_spv = common::read_spv("nearfar_compact.spv");
        compact_shader =
            device.createShaderModule({{}, compact_spv.size() * 4, compact_spv.data()});

        compact_pipeline_layout =
            device.createPipelineLayout({{}, 1, &compact_desc_bundle.layout, 1, &pcRange});

        vk::PipelineShaderStageCreateInfo compact_shader_stage{
            {}, vk::ShaderStageFlagBits::eCompute, compact_shader, "main", &spec_info};

        compact_pipeline =
            device.createComputePipeline({}, {{}, compact_shader_stage, compact_pipeline_layout})
                .value;

        std::vector<uint32_t> prepare_dispatch_spv =
            common::read_spv("nearfar_prepare_dispatch.spv");
        prepare_dispatch_shader = device.createShaderModule(
            {{}, prepare_dispatch_spv.size() * 4, prepare_dispatch_spv.data()});

        prepare_dispatch_pipeline_layout =
            device.createPipelineLayout({{}, 1, &prepare_dispatch_desc_bundle.layout});

        vk::SpecializationMapEntry prepare_dispatch_spec_entry{0, 0, sizeof(uint32_t)};
        vk::SpecializationInfo prepare_dispatch_spec_info{
            1, &prepare_dispatch_spec_entry, sizeof(uint32_t), &workgroup_size};

        vk::PipelineShaderStageCreateInfo prepare_dispatch_shader_stage{
            {},
            vk::ShaderStageFlagBits::eCompute,
            prepare_dispatch_shader,
            "main",
            &prepare_dispatch_spec_info};

        prepare_dispatch_pipeline =
            device
                .createComputePipeline(
                    {}, {{}, prepare_dispatch_shader_stage, prepare_dispatch_pipeline_layout})
                .value;
    }

    uint32_t run(vk::CommandPool &cmd_pool,
                 vk::Queue &queue,
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
                // std::cout << phase << " " << num_near << " best distance " << *gpu_best_distance
                //           << std::endl;
                cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

                for (uint32_t batch_iter = 0; batch_iter < relax_batch_size; ++batch_iter)
                {
                    uint32_t relax_desc_idx = current_near_buffer * 2 + current_far_buffer;
                    auto relax_desc_set = relax_desc_bundle.descriptor_sets[relax_desc_idx];

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
                                         prepare_dispatch_pipeline);
                    cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                               prepare_dispatch_pipeline_layout,
                                               0,
                                               prepare_dispatch_desc_bundle.descriptor_sets[0],
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

                    cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, relax_pipeline);
                    cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                               relax_pipeline_layout,
                                               0,
                                               relax_desc_set,
                                               {});

                    PushConsts pc{src_node, dst_node, num_nodes, phase, delta};
                    cmd_buf.pushConstants(relax_pipeline_layout,
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

            // std::cout << phase << " far " << num_far << " best distance " << *gpu_best_distance
            //           << std::endl;

            cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

            auto compact_desc_set = compact_desc_bundle.descriptor_sets[current_far_buffer];

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

            cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, compact_pipeline);
            cmd_buf.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, compact_pipeline_layout, 0, compact_desc_set, {});

            PushConsts pc{src_node, dst_node, num_nodes, phase, delta};
            cmd_buf.pushConstants(
                compact_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
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
            queue.submit(vk::SubmitInfo{0, nullptr, nullptr, 1, &cmd_buf});
            queue.waitIdle();

            current_far_buffer = 1 - current_far_buffer;
            current_near_buffer = 0;

            num_near = *gpu_num_near;
            num_far = *gpu_num_far;

            // std::cout << phase << " compacted to: far " << num_far << " near " << num_near
            //           << std::endl;

            common::Statistics::get().stop(common::StatisticsEvent::NEARFAR_COMPACT_DURATION,
                                           compact_start);
        }

        return *gpu_best_distance;
    }

  private:
    const GraphBuffers<GraphT> &graph_buffers;
    NearFarBuffers &nearfar_buffers;

    Statistics &statistics;

    DescriptorSetBundle relax_desc_bundle;
    vk::ShaderModule relax_shader;
    vk::Pipeline relax_pipeline;
    vk::PipelineLayout relax_pipeline_layout;

    DescriptorSetBundle compact_desc_bundle;
    vk::ShaderModule compact_shader;
    vk::Pipeline compact_pipeline;
    vk::PipelineLayout compact_pipeline_layout;

    DescriptorSetBundle prepare_dispatch_desc_bundle;
    vk::ShaderModule prepare_dispatch_shader;
    vk::Pipeline prepare_dispatch_pipeline;
    vk::PipelineLayout prepare_dispatch_pipeline_layout;

    vk::Device &device;
    uint32_t workgroup_size;
};

} // namespace gpusssp::gpu

#endif
