#ifndef GPUSSSP_GPU_DELTASETP_HPP
#define GPUSSSP_GPU_DELTASETP_HPP

#include <vulkan/vulkan.hpp>

#include "common/constants.hpp"
#include "common/shader.hpp"
#include "gpu/deltastep_buffers.hpp"
#include "gpu/graph_buffers.hpp"

#include <iostream>

namespace gpusssp::gpu
{

template <typename GraphT> class DeltaStep
{
    static constexpr const size_t WORKGROUP_SIZE = 128u;
    struct PushConsts
    {
        uint32_t src_node;
        uint32_t n;
        uint32_t bucket_idx;
        uint32_t delta;
        uint32_t iteration;
        uint32_t max_weight;
    };

  public:
    DeltaStep(const GraphBuffers<GraphT> &graph_buffers,
              DeltaStepBuffers &deltastep_buffers,
              vk::Device &device)
        : graph_buffers(graph_buffers), deltastep_buffers(deltastep_buffers), device(device)
    {
    }

    ~DeltaStep()
    {
        device.destroyShaderModule(shader);
        device.destroyPipeline(pipeline);
        device.destroyPipelineLayout(pipeline_layout);
        device.destroyDescriptorSetLayout(desc_set_layout);
        device.destroyDescriptorPool(desc_pool);
    }

    void initialize_descriptor_sets()
    {
        auto graph_bufs = graph_buffers.buffers();
        auto [dist_buffer, changed_buffer_0, changed_buffer_1] = deltastep_buffers.buffers();

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
        // Binding 4: current_changed (will swap between buffer_0 and buffer_1)
        bindings.push_back(
            {4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
        // Binding 5: previous_changed (will swap between buffer_1 and buffer_0)
        bindings.push_back(
            {5, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});

        desc_set_layout =
            device.createDescriptorSetLayout({{}, (uint32_t)bindings.size(), bindings.data()});

        // Create descriptor pool for 2 descriptor sets
        vk::DescriptorPoolSize poolSize{vk::DescriptorType::eStorageBuffer,
                                        (uint32_t)(bindings.size() * 2)};
        desc_pool = device.createDescriptorPool({{}, 2, 1, &poolSize});

        std::vector<vk::DescriptorSetLayout> layouts = {desc_set_layout, desc_set_layout};
        auto desc_sets = device.allocateDescriptorSets({desc_pool, 2, layouts.data()});
        desc_set_0 = desc_sets[0];
        desc_set_1 = desc_sets[1];

        // Update descriptor set 0: changed_buffer_0 as current, changed_buffer_1 as previous
        std::vector<vk::DescriptorBufferInfo> dbis_0;
        // Pre-allocate to avoid reallocation
        dbis_0.reserve(6);

        for (auto i = 0u; i < graph_bufs.size(); ++i)
        {
            dbis_0.push_back({graph_bufs[i], 0, VK_WHOLE_SIZE});
        }
        dbis_0.push_back({dist_buffer, 0, VK_WHOLE_SIZE});
        dbis_0.push_back({changed_buffer_0, 0, VK_WHOLE_SIZE});
        dbis_0.push_back({changed_buffer_1, 0, VK_WHOLE_SIZE});

        std::vector<vk::WriteDescriptorSet> writes_0;
        for (auto i = 0u; i < graph_bufs.size(); ++i)
        {
            writes_0.push_back({desc_set_0,
                                i,
                                0,
                                1,
                                vk::DescriptorType::eStorageBuffer,
                                nullptr,
                                &dbis_0[i],
                                nullptr});
        }
        writes_0.push_back({desc_set_0,
                            3,
                            0,
                            1,
                            vk::DescriptorType::eStorageBuffer,
                            nullptr,
                            &dbis_0[3],
                            nullptr});
        writes_0.push_back({desc_set_0,
                            4,
                            0,
                            1,
                            vk::DescriptorType::eStorageBuffer,
                            nullptr,
                            &dbis_0[4],
                            nullptr});
        writes_0.push_back({desc_set_0,
                            5,
                            0,
                            1,
                            vk::DescriptorType::eStorageBuffer,
                            nullptr,
                            &dbis_0[5],
                            nullptr});

        device.updateDescriptorSets(writes_0, {});

        // Update descriptor set 1: changed_buffer_1 as current, changed_buffer_0 as previous
        std::vector<vk::DescriptorBufferInfo> dbis_1;
        dbis_1.reserve(6);

        for (auto i = 0u; i < graph_bufs.size(); ++i)
        {
            dbis_1.push_back({graph_bufs[i], 0, VK_WHOLE_SIZE});
        }
        dbis_1.push_back({dist_buffer, 0, VK_WHOLE_SIZE});
        dbis_1.push_back({changed_buffer_1, 0, VK_WHOLE_SIZE});
        dbis_1.push_back({changed_buffer_0, 0, VK_WHOLE_SIZE});

        std::vector<vk::WriteDescriptorSet> writes_1;
        for (auto i = 0u; i < graph_bufs.size(); ++i)
        {
            writes_1.push_back({desc_set_1,
                                i,
                                0,
                                1,
                                vk::DescriptorType::eStorageBuffer,
                                nullptr,
                                &dbis_1[i],
                                nullptr});
        }
        writes_1.push_back({desc_set_1,
                            3,
                            0,
                            1,
                            vk::DescriptorType::eStorageBuffer,
                            nullptr,
                            &dbis_1[3],
                            nullptr});
        writes_1.push_back({desc_set_1,
                            4,
                            0,
                            1,
                            vk::DescriptorType::eStorageBuffer,
                            nullptr,
                            &dbis_1[4],
                            nullptr});
        writes_1.push_back({desc_set_1,
                            5,
                            0,
                            1,
                            vk::DescriptorType::eStorageBuffer,
                            nullptr,
                            &dbis_1[5],
                            nullptr});

        device.updateDescriptorSets(writes_1, {});
    }

    void initialize()
    {
        initialize_descriptor_sets();

        std::vector<uint32_t> spv = common::read_spv("delta_step.spv");
        shader = device.createShaderModule({{}, spv.size() * 4, spv.data()});

        vk::PushConstantRange pcRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConsts)};
        pipeline_layout = device.createPipelineLayout({{}, 1, &desc_set_layout, 1, &pcRange});

        vk::PipelineShaderStageCreateInfo shaderStage{
            {}, vk::ShaderStageFlagBits::eCompute, shader, "main"};

        pipeline = device.createComputePipeline({}, {{}, shaderStage, pipeline_layout}).value;
    }

    uint32_t run(vk::CommandPool &cmd_pool,
                 vk::Queue &queue,
                 uint32_t src_node,
                 uint32_t dst_node,
                 uint32_t delta)
    {
        vk::CommandBuffer cmd_buf =
            device.allocateCommandBuffers({cmd_pool, vk::CommandBufferLevel::ePrimary, 1})[0];

        const std::size_t MAX_BUCKETS = common::INF_WEIGHT / delta - 1;

        auto [gpu_num_changed_0, gpu_num_changed_1] = deltastep_buffers.num_changed();
        uint32_t *gpu_dist = deltastep_buffers.dist();
        auto num_nodes = (uint32_t)graph_buffers.num_nodes();

        std::fill(gpu_dist, gpu_dist + num_nodes, common::INF_WEIGHT);
        gpu_dist[src_node] = 0;
        uint32_t *gpu_max_dist = gpu_dist + num_nodes;
        *gpu_max_dist = 0u;

        auto [dist_buffer, changed_buffer_0, changed_buffer_1] = deltastep_buffers.buffers();

        for (uint32_t bucket = 0; bucket < MAX_BUCKETS; bucket++)
        {
            uint32_t iteration = 0;

            auto previous_changed_buffer = changed_buffer_0;
            auto current_changed_buffer = changed_buffer_1;
            auto previous_desc_set = desc_set_0;
            auto current_desc_set = desc_set_1;
            auto *previous_gpu_num_changed = gpu_num_changed_0;
            auto *current_gpu_num_changed = gpu_num_changed_1;

            do
            {
                cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
                cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);

                cmd_buf.bindDescriptorSets(
                    vk::PipelineBindPoint::eCompute, pipeline_layout, 0, current_desc_set, {});

                PushConsts pc{src_node, num_nodes, bucket, delta, iteration, delta};

                cmd_buf.fillBuffer(previous_changed_buffer, 0, VK_WHOLE_SIZE, UINT32_MAX);
                cmd_buf.fillBuffer(current_changed_buffer, 0, VK_WHOLE_SIZE, 0);
                cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                        vk::PipelineStageFlagBits::eComputeShader,
                                        vk::DependencyFlags{},
                                        vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite,
                                                          vk::AccessFlagBits::eShaderRead |
                                                              vk::AccessFlagBits::eShaderWrite},
                                        {},
                                        {});

                pc.iteration = iteration++;
                cmd_buf.pushConstants(
                    pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
                cmd_buf.dispatch((num_nodes + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, 1, 1);
                cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                        vk::PipelineStageFlagBits::eComputeShader,
                                        vk::DependencyFlags{},
                                        vk::MemoryBarrier{vk::AccessFlagBits::eShaderWrite,
                                                          vk::AccessFlagBits::eShaderRead |
                                                              vk::AccessFlagBits::eShaderWrite},
                                        {},
                                        {});

                for (unsigned i = 1; i < 64; ++i)
                {
                    std::swap(previous_changed_buffer, current_changed_buffer);
                    std::swap(previous_desc_set, current_desc_set);
                    std::swap(previous_gpu_num_changed, current_gpu_num_changed);
                    cmd_buf.bindDescriptorSets(
                        vk::PipelineBindPoint::eCompute, pipeline_layout, 0, current_desc_set, {});

                    cmd_buf.fillBuffer(current_changed_buffer, 0, VK_WHOLE_SIZE, 0);
                    cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                            vk::PipelineStageFlagBits::eComputeShader,
                                            vk::DependencyFlags{},
                                            vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite,
                                                              vk::AccessFlagBits::eShaderRead |
                                                                  vk::AccessFlagBits::eShaderWrite},
                                            {},
                                            {});

                    pc.iteration = iteration++;
                    cmd_buf.pushConstants(
                        pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
                    cmd_buf.dispatch((num_nodes + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, 1, 1);
                    cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                            vk::PipelineStageFlagBits::eComputeShader,
                                            vk::DependencyFlags{},
                                            vk::MemoryBarrier{vk::AccessFlagBits::eShaderWrite,
                                                              vk::AccessFlagBits::eShaderRead |
                                                                  vk::AccessFlagBits::eShaderWrite},
                                            {},
                                            {});
                }

                cmd_buf.end();
                queue.submit(vk::SubmitInfo{0, nullptr, nullptr, 1, &cmd_buf});
                queue.waitIdle();

                // std::cout << bucket << " " << *current_gpu_num_changed << " max distance "
                //           << *gpu_max_dist << std::endl;
                // for (auto node_id = 0u; node_id < num_nodes; ++node_id)
                //{
                //     if (gpu_dist[node_id] != common::INF_WEIGHT)
                //     {
                //         std::cout << "\t" << node_id << "\t" << gpu_dist[node_id] << std::endl;
                //     }
                // }
            } while (*current_gpu_num_changed > 0);

            cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);

            std::swap(previous_changed_buffer, current_changed_buffer);
            std::swap(previous_desc_set, current_desc_set);
            std::swap(previous_gpu_num_changed, current_gpu_num_changed);
            cmd_buf.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, pipeline_layout, 0, current_desc_set, {});

            cmd_buf.fillBuffer(previous_changed_buffer, 0, VK_WHOLE_SIZE, UINT32_MAX);
            cmd_buf.fillBuffer(current_changed_buffer, 0, VK_WHOLE_SIZE, 0);
            cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                    vk::PipelineStageFlagBits::eComputeShader,
                                    vk::DependencyFlags{},
                                    vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite,
                                                      vk::AccessFlagBits::eShaderRead |
                                                          vk::AccessFlagBits::eShaderWrite},
                                    {},
                                    {});

            PushConsts pc{src_node, num_nodes, bucket, delta, iteration++, UINT32_MAX};
            cmd_buf.pushConstants(
                pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
            cmd_buf.dispatch((num_nodes + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, 1, 1);
            cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                    vk::PipelineStageFlagBits::eComputeShader,
                                    vk::DependencyFlags{},
                                    vk::MemoryBarrier{vk::AccessFlagBits::eShaderWrite,
                                                      vk::AccessFlagBits::eShaderRead |
                                                          vk::AccessFlagBits::eShaderWrite},
                                    {},
                                    {});
            cmd_buf.end();
            queue.submit(vk::SubmitInfo{0, nullptr, nullptr, 1, &cmd_buf});
            queue.waitIdle();

            // std::cout << bucket << " heavy " << *gpu_num_changed << " max distance "
            //           << *gpu_max_dist << std::endl;

            // for (auto node_id = 0u; node_id < num_nodes; ++node_id) {
            //   if (gpu_dist[node_id] != common::INF_WEIGHT) {
            //     std::cout << "\t" << node_id << "\t" << gpu_dist[node_id] << std::endl;
            //   }
            // }

            if (gpu_dist[dst_node] != common::INF_WEIGHT)
            {
                // If the distance is smaller then the current bucket,
                // we have already settled the destination
                if (gpu_dist[dst_node] < bucket * delta)
                {
                    break;
                }
            }

            // If the maximum node distance is lower than the next bucket all other buckets
            // will be empty -> dst_node is unreachable.
            if (*gpu_max_dist < (bucket + 1) * delta)
            {
                break;
            }
        }

        return gpu_dist[dst_node];
    }

  private:
    const GraphBuffers<GraphT> &graph_buffers;
    DeltaStepBuffers &deltastep_buffers;

    vk::DescriptorSet desc_set_0;
    vk::DescriptorSet desc_set_1;
    vk::DescriptorSetLayout desc_set_layout;
    vk::DescriptorPool desc_pool;
    vk::ShaderModule shader;
    vk::Pipeline pipeline;
    vk::PipelineLayout pipeline_layout;

    vk::Device &device;
};

} // namespace gpusssp::gpu

#endif
