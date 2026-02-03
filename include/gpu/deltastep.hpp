#ifndef GPUSSSP_GPU_DELTASETP_HPP
#define GPUSSSP_GPU_DELTASETP_HPP

#include <vulkan/vulkan.hpp>

#include "common/constants.hpp"
#include "common/shader.hpp"
#include "gpu/deltastep_buffers.hpp"
#include "gpu/graph_buffers.hpp"
#include "gpu/memory.hpp"
#include "gpu/statistics.hpp"

#include <iostream>

namespace gpusssp::gpu
{

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
        device.destroyShaderModule(shader);
        device.destroyPipeline(pipeline);
        device.destroyPipelineLayout(pipeline_layout);
        device.destroyDescriptorSetLayout(desc_bundle.layout);
        device.destroyDescriptorPool(desc_bundle.pool);

        device.destroyShaderModule(prepare_dispatch_shader);
        device.destroyPipeline(prepare_dispatch_pipeline);
        device.destroyPipelineLayout(prepare_dispatch_pipeline_layout);
        device.destroyDescriptorSetLayout(prepare_dispatch_desc_bundle.layout);
        device.destroyDescriptorPool(prepare_dispatch_desc_bundle.pool);
    }

    void initialize_descriptor_sets()
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

        desc_bundle = create_descriptor_sets(device,
                                             {{first_edges_buffer,
                                               targets_buffer,
                                               weights_buffer,
                                               dist_buffer,
                                               results_buffer,
                                               changed_buffer_0,
                                               changed_buffer_1,
                                               min_max_changed_id_0,
                                               min_max_changed_id_1,
                                               statistics_buffer},
                                              {first_edges_buffer,
                                               targets_buffer,
                                               weights_buffer,
                                               dist_buffer,
                                               results_buffer,
                                               changed_buffer_1,
                                               changed_buffer_0,
                                               min_max_changed_id_1,
                                               min_max_changed_id_0,
                                               statistics_buffer}});
    }

    void initialize_prepare_dispatch_descriptor_sets()
    {
        auto [dist_buffer,
              results_buffer,
              changed_buffer_0,
              changed_buffer_1,
              min_max_changed_id_0,
              min_max_changed_id_1,
              dispatch_buffer] = deltastep_buffers.buffers();

        prepare_dispatch_desc_bundle = create_descriptor_sets(
            device,
            {{min_max_changed_id_1, dispatch_buffer}, {min_max_changed_id_0, dispatch_buffer}});
    }

    void initialize()
    {
        initialize_descriptor_sets();
        initialize_prepare_dispatch_descriptor_sets();

        std::vector<uint32_t> spv = common::read_spv("delta_step.spv");
        shader = device.createShaderModule({{}, spv.size() * 4, spv.data()});

        vk::PushConstantRange pcRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConsts)};
        pipeline_layout = device.createPipelineLayout({{}, 1, &desc_bundle.layout, 1, &pcRange});

        // Setup specialization constant for workgroup size
        vk::SpecializationMapEntry spec_entry{0, 0, sizeof(uint32_t)};
        vk::SpecializationInfo spec_info{1, &spec_entry, sizeof(uint32_t), &workgroup_size};

        vk::PipelineShaderStageCreateInfo shaderStage{
            {}, vk::ShaderStageFlagBits::eCompute, shader, "main", &spec_info};

        pipeline = device.createComputePipeline({}, {{}, shaderStage, pipeline_layout}).value;

        std::vector<uint32_t> prepare_dispatch_spv =
            common::read_spv("deltastep_prepare_dispatch.spv");
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
                 uint32_t batch_size = 64)
    {
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

        // Initialize dist buffer on GPU
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

        for (uint32_t bucket = 0; bucket < MAX_BUCKETS; bucket++)
        {
            PushConsts pc{src_node, dst_node, num_nodes, bucket, delta, delta};

            auto previous_changed_buffer = changed_buffer_0;
            auto current_changed_buffer = changed_buffer_1;
            auto previous_min_max_changed_id_buffer = min_max_changed_id_buffer_0;
            auto current_min_max_changed_id_buffer = min_max_changed_id_buffer_1;
            auto *gpu_prev_min_changed_id = gpu_min_max_changed_id_0;
            auto *gpu_current_min_changed_id = gpu_min_max_changed_id_1;
            auto *gpu_prev_max_changed_id = gpu_min_max_changed_id_0 + 1;
            auto *gpu_current_max_changed_id = gpu_min_max_changed_id_1 + 1;
            auto prev_dispatch_desc_set = prepare_dispatch_desc_bundle.descriptor_sets[0];
            auto current_dispatch_desc_set = prepare_dispatch_desc_bundle.descriptor_sets[1];
            auto prev_desc_set = desc_bundle.descriptor_sets[0];
            auto current_desc_set = desc_bundle.descriptor_sets[1];

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
                                         prepare_dispatch_pipeline);
                    cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                               prepare_dispatch_pipeline_layout,
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

                    cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
                    cmd_buf.bindDescriptorSets(
                        vk::PipelineBindPoint::eCompute, pipeline_layout, 0, current_desc_set, {});

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

                    cmd_buf.pushConstants(
                        pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
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
                queue.submit(vk::SubmitInfo{0, nullptr, nullptr, 1, &cmd_buf});
                queue.waitIdle();

                // std::cout << bucket << " changed " << *gpu_prev_min_changed_id << "-"
                //           << *gpu_prev_max_changed_id << " max " << *gpu_max_distance << " best "
                //           << *gpu_best_distance << std::endl;

                // start a new command buffer either for next iteration here or the heavy pass
                cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

                // since we do this after the swap we need to look at prev not current
            } while (*gpu_prev_min_changed_id < num_nodes);

            cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
            cmd_buf.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, pipeline_layout, 0, current_desc_set, {});

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
                pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
            cmd_buf.dispatch((num_nodes + workgroup_size - 1) / workgroup_size, 1, 1);
            cmd_buf.pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader,
                vk::PipelineStageFlagBits::eHost,
                vk::DependencyFlags{},
                vk::MemoryBarrier{vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead},
                {},
                {});
            cmd_buf.end();
            queue.submit(vk::SubmitInfo{0, nullptr, nullptr, 1, &cmd_buf});
            queue.waitIdle();

            // std::cout << bucket << " heavy changed " << *gpu_current_min_changed_id << "-"
            //           << *gpu_current_max_changed_id << " max " << *gpu_max_distance << " best "
            //           << *gpu_best_distance << std::endl;

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

        return *gpu_best_distance;
    }

  private:
    const GraphBuffers<GraphT> &graph_buffers;
    DeltaStepBuffers &deltastep_buffers;
    Statistics &statistics;

    DescriptorSetBundle desc_bundle;
    vk::ShaderModule shader;
    vk::Pipeline pipeline;
    vk::PipelineLayout pipeline_layout;

    DescriptorSetBundle prepare_dispatch_desc_bundle;
    vk::ShaderModule prepare_dispatch_shader;
    vk::Pipeline prepare_dispatch_pipeline;
    vk::PipelineLayout prepare_dispatch_pipeline_layout;

    vk::Device &device;
    uint32_t workgroup_size;
};

} // namespace gpusssp::gpu

#endif
