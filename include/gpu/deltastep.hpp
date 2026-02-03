#ifndef GPUSSSP_GPU_DELTASETP_HPP
#define GPUSSSP_GPU_DELTASETP_HPP

#include <vulkan/vulkan.hpp>

#include "common/constants.hpp"
#include "common/shader.hpp"
#include "gpu/deltastep_buffers.hpp"
#include "gpu/graph_buffers.hpp"
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
    struct PrepareDispatchPushConsts
    {
        uint32_t workgroup_size;
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
        device.destroyDescriptorSetLayout(desc_set_layout);
        device.destroyDescriptorPool(desc_pool);

        device.destroyShaderModule(prepare_dispatch_shader);
        device.destroyPipeline(prepare_dispatch_pipeline);
        device.destroyPipelineLayout(prepare_dispatch_pipeline_layout);
        device.destroyDescriptorSetLayout(prepare_dispatch_desc_set_layout);
        device.destroyDescriptorPool(prepare_dispatch_desc_pool);
    }

    void initialize_descriptor_sets()
    {
        auto graph_bufs = graph_buffers.buffers();
        auto [dist_buffer,
              results_buffer,
              changed_buffer_0,
              changed_buffer_1,
              min_max_changed_id_0,
              min_max_changed_id_1,
              dispatch_buffer] = deltastep_buffers.buffers();
        auto statistics_buffer = statistics.buffer();

        // Create bindings for all buffers
        std::vector<vk::DescriptorSetLayoutBinding> bindings;
        // Bindings 0-2: Graph buffers (first_edges, targets, weight)
        for (auto i = 0u; i < graph_bufs.size(); i++)
        {
            bindings.push_back(
                {i, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
        }
        // Binding 3: dist buffer
        bindings.push_back(
            {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
        // Binding 4: results buffer (best_distance, max_distance)
        bindings.push_back(
            {4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
        // Binding 5: current_changed (will swap between buffer_0 and buffer_1)
        bindings.push_back(
            {5, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
        // Binding 6: previous_changed (will swap between buffer_1 and buffer_0)
        bindings.push_back(
            {6, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
        // Binding 7: current min/max changed id buffer
        bindings.push_back(
            {7, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
        // Binding 8: previous min/max changed id buffer
        bindings.push_back(
            {8, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
        // Binding 9: statistics counters
        bindings.push_back(
            {9, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});

        desc_set_layout =
            device.createDescriptorSetLayout({{}, (uint32_t)bindings.size(), bindings.data()});

        // Create descriptor pool for 2 descriptor sets
        vk::DescriptorPoolSize poolSize{vk::DescriptorType::eStorageBuffer,
                                        (uint32_t)(bindings.size() * 2)};
        desc_pool = device.createDescriptorPool({{}, 2, 1, &poolSize});

        std::vector<vk::DescriptorSetLayout> layouts = {desc_set_layout, desc_set_layout};
        desc_sets = device.allocateDescriptorSets({desc_pool, 2, layouts.data()});

        for (uint32_t i = 0; i < 2; ++i)
        {
            auto current_changed = i == 0 ? changed_buffer_0 : changed_buffer_1;
            auto previous_changed = i == 0 ? changed_buffer_1 : changed_buffer_0;
            auto current_min_max = i == 0 ? min_max_changed_id_0 : min_max_changed_id_1;
            auto previous_min_max = i == 0 ? min_max_changed_id_1 : min_max_changed_id_0;

            std::vector<vk::DescriptorBufferInfo> dbis;
            dbis.reserve(10);

            for (auto j = 0u; j < graph_bufs.size(); ++j)
            {
                dbis.push_back({graph_bufs[j], 0, VK_WHOLE_SIZE});
            }
            dbis.push_back({dist_buffer, 0, VK_WHOLE_SIZE});
            dbis.push_back({results_buffer, 0, VK_WHOLE_SIZE});
            dbis.push_back({current_changed, 0, VK_WHOLE_SIZE});
            dbis.push_back({previous_changed, 0, VK_WHOLE_SIZE});
            dbis.push_back({current_min_max, 0, VK_WHOLE_SIZE});
            dbis.push_back({previous_min_max, 0, VK_WHOLE_SIZE});
            dbis.push_back({statistics_buffer, 0, VK_WHOLE_SIZE});

            std::vector<vk::WriteDescriptorSet> writes;
            for (auto j = 0u; j < dbis.size(); ++j)
            {
                writes.push_back({desc_sets[i],
                                  j,
                                  0,
                                  1,
                                  vk::DescriptorType::eStorageBuffer,
                                  nullptr,
                                  &dbis[j],
                                  nullptr});
            }

            device.updateDescriptorSets(writes, {});
        }
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

        std::vector<vk::DescriptorSetLayoutBinding> bindings;
        bindings.push_back(
            {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
        bindings.push_back(
            {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});

        prepare_dispatch_desc_set_layout =
            device.createDescriptorSetLayout({{}, (uint32_t)bindings.size(), bindings.data()});

        vk::DescriptorPoolSize poolSize{vk::DescriptorType::eStorageBuffer,
                                        (uint32_t)(bindings.size() * 2)};
        prepare_dispatch_desc_pool = device.createDescriptorPool({{}, 2, 1, &poolSize});

        std::vector<vk::DescriptorSetLayout> layouts = {prepare_dispatch_desc_set_layout,
                                                        prepare_dispatch_desc_set_layout};
        prepare_dispatch_desc_sets =
            device.allocateDescriptorSets({prepare_dispatch_desc_pool, 2, layouts.data()});

        for (uint32_t i = 0; i < 2; ++i)
        {
            auto min_max_buffer = i == 0 ? min_max_changed_id_1 : min_max_changed_id_0;

            std::vector<vk::DescriptorBufferInfo> dbis;
            dbis.push_back({min_max_buffer, 0, VK_WHOLE_SIZE});
            dbis.push_back({dispatch_buffer, 0, VK_WHOLE_SIZE});

            std::vector<vk::WriteDescriptorSet> writes;
            for (auto j = 0u; j < dbis.size(); ++j)
            {
                writes.push_back({prepare_dispatch_desc_sets[i],
                                  j,
                                  0,
                                  1,
                                  vk::DescriptorType::eStorageBuffer,
                                  nullptr,
                                  &dbis[j],
                                  nullptr});
            }
            device.updateDescriptorSets(writes, {});
        }
    }

    void initialize()
    {
        initialize_descriptor_sets();
        initialize_prepare_dispatch_descriptor_sets();

        std::vector<uint32_t> spv = common::read_spv("delta_step.spv");
        shader = device.createShaderModule({{}, spv.size() * 4, spv.data()});

        vk::PushConstantRange pcRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConsts)};
        pipeline_layout = device.createPipelineLayout({{}, 1, &desc_set_layout, 1, &pcRange});

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

        vk::PushConstantRange prepare_dispatch_pcRange{
            vk::ShaderStageFlagBits::eCompute, 0, sizeof(PrepareDispatchPushConsts)};
        prepare_dispatch_pipeline_layout = device.createPipelineLayout(
            {{}, 1, &prepare_dispatch_desc_set_layout, 1, &prepare_dispatch_pcRange});

        vk::PipelineShaderStageCreateInfo prepare_dispatch_shader_stage{
            {}, vk::ShaderStageFlagBits::eCompute, prepare_dispatch_shader, "main", nullptr};

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
            auto prev_dispatch_desc_set = prepare_dispatch_desc_sets[0];
            auto current_dispatch_desc_set = prepare_dispatch_desc_sets[1];
            auto prev_desc_set = desc_sets[0];
            auto current_desc_set = desc_sets[1];

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
                    PrepareDispatchPushConsts prepare_pc{workgroup_size};
                    cmd_buf.pushConstants(prepare_dispatch_pipeline_layout,
                                          vk::ShaderStageFlagBits::eCompute,
                                          0,
                                          sizeof(prepare_pc),
                                          &prepare_pc);
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

    std::vector<vk::DescriptorSet> desc_sets;
    vk::DescriptorSetLayout desc_set_layout;
    vk::DescriptorPool desc_pool;
    vk::ShaderModule shader;
    vk::Pipeline pipeline;
    vk::PipelineLayout pipeline_layout;

    std::vector<vk::DescriptorSet> prepare_dispatch_desc_sets;
    vk::DescriptorSetLayout prepare_dispatch_desc_set_layout;
    vk::DescriptorPool prepare_dispatch_desc_pool;
    vk::ShaderModule prepare_dispatch_shader;
    vk::Pipeline prepare_dispatch_pipeline;
    vk::PipelineLayout prepare_dispatch_pipeline_layout;

    vk::Device &device;
    uint32_t workgroup_size;
};

} // namespace gpusssp::gpu

#endif
