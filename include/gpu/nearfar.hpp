#ifndef GPUSSSP_GPU_NEARFAR_HPP
#define GPUSSSP_GPU_NEARFAR_HPP

#include <vulkan/vulkan.hpp>

#include "common/constants.hpp"
#include "common/shader.hpp"
#include "gpu/graph_buffers.hpp"
#include "gpu/nearfar_buffers.hpp"

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
    struct PrepareDispatchPushConsts
    {
        uint32_t workgroup_size;
        uint32_t counter_index;
    };

  public:
    NearFar(const GraphBuffers<GraphT> &graph_buffers,
            NearFarBuffers &nearfar_buffers,
            vk::Device &device,
            uint32_t workgroup_size = DEFAULT_WORKGROUP_SIZE)
        : graph_buffers(graph_buffers), nearfar_buffers(nearfar_buffers), device(device),
          workgroup_size(workgroup_size)
    {
    }

    ~NearFar()
    {
        device.destroyShaderModule(relax_shader);
        device.destroyPipeline(relax_pipeline);
        device.destroyPipelineLayout(relax_pipeline_layout);
        device.destroyDescriptorSetLayout(relax_desc_set_layout);
        device.destroyDescriptorPool(relax_desc_pool);

        device.destroyShaderModule(compact_shader);
        device.destroyPipeline(compact_pipeline);
        device.destroyPipelineLayout(compact_pipeline_layout);
        device.destroyDescriptorSetLayout(compact_desc_set_layout);
        device.destroyDescriptorPool(compact_desc_pool);

        device.destroyShaderModule(prepare_dispatch_shader);
        device.destroyPipeline(prepare_dispatch_pipeline);
        device.destroyPipelineLayout(prepare_dispatch_pipeline_layout);
        device.destroyDescriptorSetLayout(prepare_dispatch_desc_set_layout);
        device.destroyDescriptorPool(prepare_dispatch_desc_pool);
    }

    void initialize_relax_descriptor_sets()
    {
        auto graph_bufs = graph_buffers.buffers();
        auto [dist_buffer,
              results_buffer,
              near_0_buffer,
              near_1_buffer,
              far_0_buffer,
              far_1_buffer,
              counters_buffer,
              dispatch_relax_buffer] = nearfar_buffers.buffers();

        std::vector<vk::DescriptorSetLayoutBinding> bindings;
        for (auto i = 0u; i < graph_bufs.size(); i++)
        {
            bindings.push_back(
                {i, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
        }
        bindings.push_back(
            {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
        bindings.push_back(
            {4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
        bindings.push_back(
            {5, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
        bindings.push_back(
            {6, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
        bindings.push_back(
            {7, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
        bindings.push_back(
            {8, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});

        relax_desc_set_layout =
            device.createDescriptorSetLayout({{}, (uint32_t)bindings.size(), bindings.data()});

        vk::DescriptorPoolSize poolSize{vk::DescriptorType::eStorageBuffer,
                                        (uint32_t)(bindings.size() * 4)};
        relax_desc_pool = device.createDescriptorPool({{}, 4, 1, &poolSize});

        std::vector<vk::DescriptorSetLayout> layouts = {relax_desc_set_layout,
                                                        relax_desc_set_layout,
                                                        relax_desc_set_layout,
                                                        relax_desc_set_layout};
        auto desc_sets = device.allocateDescriptorSets({relax_desc_pool, 4, layouts.data()});
        for (size_t i = 0; i < 4; ++i)
        {
            relax_desc_sets[i] = desc_sets[i];
        }

        for (uint32_t current_near_buffer = 0; current_near_buffer < 2; ++current_near_buffer)
        {
            for (uint32_t current_far_buffer = 0; current_far_buffer < 2; ++current_far_buffer)
            {
                uint32_t desc_idx = current_near_buffer * 2 + current_far_buffer;
                auto desc_set = relax_desc_sets[desc_idx];

                auto near_buffer = current_near_buffer == 0 ? near_0_buffer : near_1_buffer;
                auto next_near_buffer = current_near_buffer == 0 ? near_1_buffer : near_0_buffer;
                auto far_buffer = current_far_buffer == 0 ? far_0_buffer : far_1_buffer;

                std::vector<vk::DescriptorBufferInfo> dbis;
                dbis.reserve(9);

                for (auto i = 0u; i < graph_bufs.size(); ++i)
                {
                    dbis.push_back({graph_bufs[i], 0, VK_WHOLE_SIZE});
                }
                dbis.push_back({dist_buffer, 0, VK_WHOLE_SIZE});
                dbis.push_back({results_buffer, 0, VK_WHOLE_SIZE});
                dbis.push_back({near_buffer, 0, VK_WHOLE_SIZE});
                dbis.push_back({next_near_buffer, 0, VK_WHOLE_SIZE});
                dbis.push_back({far_buffer, 0, VK_WHOLE_SIZE});
                dbis.push_back({counters_buffer, 0, VK_WHOLE_SIZE});

                std::vector<vk::WriteDescriptorSet> writes;
                for (auto i = 0u; i < dbis.size(); ++i)
                {
                    writes.push_back({desc_set,
                                      i,
                                      0,
                                      1,
                                      vk::DescriptorType::eStorageBuffer,
                                      nullptr,
                                      &dbis[i],
                                      nullptr});
                }

                device.updateDescriptorSets(writes, {});
            }
        }
    }

    void initialize_compact_descriptor_sets()
    {
        auto graph_bufs = graph_buffers.buffers();
        auto [dist_buffer,
              results_buffer,
              near_0_buffer,
              near_1_buffer,
              far_0_buffer,
              far_1_buffer,
              counters_buffer,
              dispatch_relax_buffer] = nearfar_buffers.buffers();

        std::vector<vk::DescriptorSetLayoutBinding> bindings;
        for (auto i = 0u; i < graph_bufs.size(); i++)
        {
            bindings.push_back(
                {i, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
        }
        bindings.push_back(
            {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
        bindings.push_back(
            {4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
        bindings.push_back(
            {5, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
        bindings.push_back(
            {6, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
        bindings.push_back(
            {7, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
        bindings.push_back(
            {8, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});

        compact_desc_set_layout =
            device.createDescriptorSetLayout({{}, (uint32_t)bindings.size(), bindings.data()});

        vk::DescriptorPoolSize poolSize{vk::DescriptorType::eStorageBuffer,
                                        (uint32_t)(bindings.size() * 2)};
        compact_desc_pool = device.createDescriptorPool({{}, 2, 1, &poolSize});

        std::vector<vk::DescriptorSetLayout> layouts = {compact_desc_set_layout,
                                                        compact_desc_set_layout};
        auto desc_sets = device.allocateDescriptorSets({compact_desc_pool, 2, layouts.data()});
        for (size_t i = 0; i < 2; ++i)
        {
            compact_desc_sets[i] = desc_sets[i];
        }

        for (uint32_t current_far_buffer = 0; current_far_buffer < 2; ++current_far_buffer)
        {
            auto desc_set = compact_desc_sets[current_far_buffer];

            auto far_buffer = current_far_buffer == 0 ? far_0_buffer : far_1_buffer;
            auto next_far_buffer = current_far_buffer == 0 ? far_1_buffer : far_0_buffer;

            std::vector<vk::DescriptorBufferInfo> dbis;
            dbis.reserve(9);

            for (auto i = 0u; i < graph_bufs.size(); ++i)
            {
                dbis.push_back({graph_bufs[i], 0, VK_WHOLE_SIZE});
            }
            dbis.push_back({dist_buffer, 0, VK_WHOLE_SIZE});
            dbis.push_back({results_buffer, 0, VK_WHOLE_SIZE});
            dbis.push_back({far_buffer, 0, VK_WHOLE_SIZE});
            dbis.push_back({near_0_buffer, 0, VK_WHOLE_SIZE});
            dbis.push_back({next_far_buffer, 0, VK_WHOLE_SIZE});
            dbis.push_back({counters_buffer, 0, VK_WHOLE_SIZE});

            std::vector<vk::WriteDescriptorSet> writes;
            for (auto i = 0u; i < dbis.size(); ++i)
            {
                writes.push_back({desc_set,
                                  i,
                                  0,
                                  1,
                                  vk::DescriptorType::eStorageBuffer,
                                  nullptr,
                                  &dbis[i],
                                  nullptr});
            }

            device.updateDescriptorSets(writes, {});
        }
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
              dispatch_relax_buffer] = nearfar_buffers.buffers();

        std::vector<vk::DescriptorSetLayoutBinding> bindings;
        bindings.push_back(
            {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
        bindings.push_back(
            {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});

        prepare_dispatch_desc_set_layout =
            device.createDescriptorSetLayout({{}, (uint32_t)bindings.size(), bindings.data()});

        vk::DescriptorPoolSize poolSize{vk::DescriptorType::eStorageBuffer,
                                        (uint32_t)(bindings.size())};
        prepare_dispatch_desc_pool = device.createDescriptorPool({{}, 1, 1, &poolSize});

        prepare_dispatch_desc_set = device.allocateDescriptorSets(
            {prepare_dispatch_desc_pool, 1, &prepare_dispatch_desc_set_layout})[0];

        std::vector<vk::DescriptorBufferInfo> dbis;
        dbis.push_back({counters_buffer, 0, VK_WHOLE_SIZE});
        dbis.push_back({dispatch_relax_buffer, 0, VK_WHOLE_SIZE});

        std::vector<vk::WriteDescriptorSet> writes;
        for (auto i = 0u; i < dbis.size(); ++i)
        {
            writes.push_back({prepare_dispatch_desc_set,
                              i,
                              0,
                              1,
                              vk::DescriptorType::eStorageBuffer,
                              nullptr,
                              &dbis[i],
                              nullptr});
        }

        device.updateDescriptorSets(writes, {});
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
            device.createPipelineLayout({{}, 1, &relax_desc_set_layout, 1, &pcRange});

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
            device.createPipelineLayout({{}, 1, &compact_desc_set_layout, 1, &pcRange});

        vk::PipelineShaderStageCreateInfo compact_shader_stage{
            {}, vk::ShaderStageFlagBits::eCompute, compact_shader, "main", &spec_info};

        compact_pipeline =
            device.createComputePipeline({}, {{}, compact_shader_stage, compact_pipeline_layout})
                .value;

        std::vector<uint32_t> prepare_dispatch_spv =
            common::read_spv("nearfar_prepare_dispatch.spv");
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
                 uint32_t relax_batch_size = 4)
    {
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
              dispatch_relax_buffer] = nearfar_buffers.buffers();

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

        while (true)
        {
            while (num_near > 0)
            {
                // std::cout << phase << " " << num_near << " best distance " << *gpu_best_distance
                //           << std::endl;
                cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

                for (uint32_t batch_iter = 0; batch_iter < relax_batch_size; ++batch_iter)
                {
                    uint32_t relax_desc_idx = current_near_buffer * 2 + current_far_buffer;
                    auto relax_desc_set = relax_desc_sets[relax_desc_idx];

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
                                               prepare_dispatch_desc_set,
                                               {});

                    PrepareDispatchPushConsts prepare_pc{workgroup_size, 0};
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

            if (*gpu_best_distance != common::INF_WEIGHT)
            {
                if (*gpu_best_distance < phase * delta)
                {
                    break;
                }
            }

            phase++;

            num_far = *gpu_num_far;
            if (num_far == 0)
            {
                break;
            }

            // std::cout << phase << " far " << num_far << " best distance " << *gpu_best_distance
            //           << std::endl;

            cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

            auto compact_desc_set = compact_desc_sets[current_far_buffer];

            cmd_buf.fillBuffer(counters_buffer, 0 * sizeof(uint32_t), sizeof(uint32_t), 0);
            cmd_buf.fillBuffer(counters_buffer, 3 * sizeof(uint32_t), sizeof(uint32_t), 0);
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
        }

        return *gpu_best_distance;
    }

  private:
    const GraphBuffers<GraphT> &graph_buffers;
    NearFarBuffers &nearfar_buffers;

    std::array<vk::DescriptorSet, 4> relax_desc_sets;
    vk::DescriptorSetLayout relax_desc_set_layout;
    vk::DescriptorPool relax_desc_pool;
    vk::ShaderModule relax_shader;
    vk::Pipeline relax_pipeline;
    vk::PipelineLayout relax_pipeline_layout;

    std::array<vk::DescriptorSet, 2> compact_desc_sets;
    vk::DescriptorSetLayout compact_desc_set_layout;
    vk::DescriptorPool compact_desc_pool;
    vk::ShaderModule compact_shader;
    vk::Pipeline compact_pipeline;
    vk::PipelineLayout compact_pipeline_layout;

    vk::DescriptorSet prepare_dispatch_desc_set;
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
