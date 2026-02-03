#ifndef GPUSSSP_GPU_BELLMANFORD_HPP
#define GPUSSSP_GPU_BELLMANFORD_HPP

#include <vulkan/vulkan.hpp>

#include "common/constants.hpp"
#include "common/shader.hpp"
#include "gpu/bellmanford_buffers.hpp"
#include "gpu/graph_buffers.hpp"
#include "gpu/memory.hpp"
#include "gpu/statistics.hpp"

#include <iostream>

namespace gpusssp::gpu
{

template <typename GraphT> class BellmanFord
{
    static constexpr const size_t DEFAULT_WORKGROUP_SIZE = 128u;
    struct PushConsts
    {
        uint32_t src_node;
        uint32_t dst_node;
        uint32_t n;
    };

  public:
    BellmanFord(const GraphBuffers<GraphT> &graph_buffers,
                BellmanFordBuffers &bellmanford_buffers,
                vk::Device &device,
                Statistics &statistics,
                uint32_t workgroup_size = DEFAULT_WORKGROUP_SIZE)
        : graph_buffers(graph_buffers), bellmanford_buffers(bellmanford_buffers),
          statistics(statistics), device(device), workgroup_size(workgroup_size)
    {
    }

    ~BellmanFord()
    {
        device.destroyShaderModule(shader);
        device.destroyPipeline(pipeline);
        device.destroyPipelineLayout(pipeline_layout);
        device.destroyDescriptorSetLayout(desc_bundle.layout);
        device.destroyDescriptorPool(desc_bundle.pool);
    }

    void initialize_descriptor_sets()
    {
        auto [first_edges_buffer, targets_buffer, weights_buffer] = graph_buffers.buffers();
        auto [dist_buffer, results_buffer, changed_buffer] = bellmanford_buffers.buffers();
        auto statistics_buffer = statistics.buffer();

        desc_bundle = create_descriptor_sets(device,
                                             {{first_edges_buffer,
                                               targets_buffer,
                                               weights_buffer,
                                               dist_buffer,
                                               results_buffer,
                                               changed_buffer,
                                               statistics_buffer}});
    }

    void initialize()
    {
        initialize_descriptor_sets();

        std::vector<uint32_t> spv = common::read_spv("bellman_ford.spv");
        shader = device.createShaderModule({{}, spv.size() * 4, spv.data()});

        vk::PushConstantRange pcRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConsts)};
        pipeline_layout = device.createPipelineLayout({{}, 1, &desc_bundle.layout, 1, &pcRange});

        // Setup specialization constant for workgroup size
        vk::SpecializationMapEntry spec_entry{0, 0, sizeof(uint32_t)};
        vk::SpecializationInfo spec_info{1, &spec_entry, sizeof(uint32_t), &workgroup_size};

        vk::PipelineShaderStageCreateInfo shaderStage{
            {}, vk::ShaderStageFlagBits::eCompute, shader, "main", &spec_info};

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

        const constexpr size_t BATCH_SIZE = 32;
        for (uint32_t iteration = 0; iteration < num_nodes - 1; iteration += BATCH_SIZE)
        {
            cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
            cmd_buf.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, pipeline_layout, 0, desc_bundle.descriptor_sets[0], {});

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
                cmd_buf.dispatch((num_nodes + workgroup_size - 1) / workgroup_size, 1, 1);
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
    Statistics &statistics;

    DescriptorSetBundle desc_bundle;
    vk::ShaderModule shader;
    vk::Pipeline pipeline;
    vk::PipelineLayout pipeline_layout;

    vk::Device &device;
    uint32_t workgroup_size;
};

} // namespace gpusssp::gpu

#endif
