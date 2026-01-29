#ifndef GPUSSSP_GPU_BELLMANFORD_HPP
#define GPUSSSP_GPU_BELLMANFORD_HPP

#include <vulkan/vulkan.hpp>

#include "common/constants.hpp"
#include "common/shader.hpp"
#include "gpu/bellmanford_buffers.hpp"
#include "gpu/graph_buffers.hpp"

#include <iostream>

namespace gpusssp::gpu
{

template <typename GraphT> class BellmanFord
{
    static constexpr const size_t WORKGROUP_SIZE = 128u;
    struct PushConsts
    {
        uint32_t src_node;
        uint32_t dst_node;
        uint32_t n;
    };

  public:
    BellmanFord(const GraphBuffers<GraphT> &graph_buffers,
                BellmanFordBuffers &bellmanford_buffers,
                vk::Device &device)
        : graph_buffers(graph_buffers), bellmanford_buffers(bellmanford_buffers), device(device)
    {
    }

    ~BellmanFord()
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
        auto [dist_buffer, results_buffer, changed_buffer] = bellmanford_buffers.buffers();

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
        // Binding 4: results buffer (best_distance)
        bindings.push_back(
            {4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
        // Binding 5: changed flag
        bindings.push_back(
            {5, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});

        desc_set_layout =
            device.createDescriptorSetLayout({{}, (uint32_t)bindings.size(), bindings.data()});

        // Create descriptor pool
        vk::DescriptorPoolSize poolSize{vk::DescriptorType::eStorageBuffer,
                                        (uint32_t)bindings.size()};
        desc_pool = device.createDescriptorPool({{}, 1, 1, &poolSize});

        auto desc_sets = device.allocateDescriptorSets({desc_pool, 1, &desc_set_layout});
        desc_set = desc_sets[0];

        // Update descriptor set
        std::vector<vk::DescriptorBufferInfo> dbis;
        dbis.reserve(6);

        for (auto i = 0u; i < graph_bufs.size(); ++i)
        {
            dbis.push_back({graph_bufs[i], 0, VK_WHOLE_SIZE});
        }
        dbis.push_back({dist_buffer, 0, VK_WHOLE_SIZE});
        dbis.push_back({results_buffer, 0, VK_WHOLE_SIZE});
        dbis.push_back({changed_buffer, 0, VK_WHOLE_SIZE});

        std::vector<vk::WriteDescriptorSet> writes;
        for (auto i = 0u; i < graph_bufs.size(); ++i)
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
        writes.push_back(
            {desc_set, 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &dbis[3], nullptr});
        writes.push_back(
            {desc_set, 4, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &dbis[4], nullptr});
        writes.push_back(
            {desc_set, 5, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &dbis[5], nullptr});

        device.updateDescriptorSets(writes, {});
    }

    void initialize()
    {
        initialize_descriptor_sets();

        std::vector<uint32_t> spv = common::read_spv("bellman_ford.spv");
        shader = device.createShaderModule({{}, spv.size() * 4, spv.data()});

        vk::PushConstantRange pcRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConsts)};
        pipeline_layout = device.createPipelineLayout({{}, 1, &desc_set_layout, 1, &pcRange});

        vk::PipelineShaderStageCreateInfo shaderStage{
            {}, vk::ShaderStageFlagBits::eCompute, shader, "main"};

        pipeline = device.createComputePipeline({}, {{}, shaderStage, pipeline_layout}).value;
    }

    uint32_t run(vk::CommandPool &cmd_pool, vk::Queue &queue, uint32_t src_node, uint32_t dst_node)
    {
        vk::CommandBuffer cmd_buf =
            device.allocateCommandBuffers({cmd_pool, vk::CommandBufferLevel::ePrimary, 1})[0];

        uint32_t *gpu_changed = bellmanford_buffers.changed();
        uint32_t *gpu_best_distance = bellmanford_buffers.best_distance();
        auto num_nodes = (uint32_t)graph_buffers.num_nodes();

        *gpu_best_distance = common::INF_WEIGHT;

        auto [dist_buffer, results_buffer, changed_buffer] = bellmanford_buffers.buffers();

        // Initialize dist buffer on GPU
        cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        if (src_node > 0)
        {
            cmd_buf.fillBuffer(dist_buffer, 0, src_node * sizeof(uint32_t), common::INF_WEIGHT);
        }
        cmd_buf.fillBuffer(
            dist_buffer, src_node * sizeof(uint32_t), (src_node + 1) * sizeof(uint32_t), 0);
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

        const constexpr size_t BATCH_SIZE = 32;
        for (uint32_t iteration = 0; iteration < num_nodes - 1; iteration += BATCH_SIZE)
        {
            cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
            cmd_buf.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, pipeline_layout, 0, desc_set, {});

            for (unsigned i = 0; i < BATCH_SIZE; ++i)
            {
                cmd_buf.fillBuffer(changed_buffer, 0, VK_WHOLE_SIZE, 0);
                cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                        vk::PipelineStageFlagBits::eComputeShader,
                                        vk::DependencyFlags{},
                                        vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite,
                                                          vk::AccessFlagBits::eShaderRead |
                                                              vk::AccessFlagBits::eShaderWrite},
                                        {},
                                        {});

                PushConsts pc{src_node, dst_node, num_nodes};
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

            // Early termination if no changes
            if (*gpu_changed == 0)
            {
                break;
            }
        }

        return *gpu_best_distance;
    }

  private:
    const GraphBuffers<GraphT> &graph_buffers;
    BellmanFordBuffers &bellmanford_buffers;

    vk::DescriptorSet desc_set;
    vk::DescriptorSetLayout desc_set_layout;
    vk::DescriptorPool desc_pool;
    vk::ShaderModule shader;
    vk::Pipeline pipeline;
    vk::PipelineLayout pipeline_layout;

    vk::Device &device;
};

} // namespace gpusssp::gpu

#endif
