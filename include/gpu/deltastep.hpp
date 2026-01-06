#ifndef GPUSSSP_GPU_DELTASETP_HPP
#define GPUSSSP_GPU_DELTASETP_HPP

#include <vulkan/vulkan.hpp>

#include "common/shader.hpp"
#include "gpu/deltastep_buffers.hpp"
#include "gpu/graph_buffers.hpp"

namespace gpusssp::gpu {

template<typename GraphT>
class DeltaStep {
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

  DeltaStep(const GraphBuffers<GraphT>& graph_buffers, DeltaStepBuffers& deltastep_buffers, vk::Device& device)
    : graph_buffers(graph_buffers), deltastep_buffers(deltastep_buffers), device(device)
  {
  }

  ~DeltaStep() {
    device.destroyShaderModule(shader);
    device.destroyPipeline(pipeline);
    device.destroyPipelineLayout(pipeline_layout);
    device.destroyDescriptorSetLayout(desc_set_layout);
    device.destroyDescriptorPool(desc_pool);
  }



  void initialize_descriptor_set() {
    std::vector<vk::Buffer> buffers;
    for (auto buffer : graph_buffers.buffers()) {
      buffers.push_back(buffer);
    }
    for (auto buffer : deltastep_buffers.buffers()) {
      buffers.push_back(buffer);
    }

    std::vector<vk::DescriptorSetLayoutBinding> bindings(buffers.size());
    for (auto i = 0u; i < buffers.size(); i++)
    {
        bindings[i].binding = i;
        bindings[i].descriptorType = vk::DescriptorType::eStorageBuffer;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = vk::ShaderStageFlagBits::eCompute;
    }
    desc_set_layout =
        device.createDescriptorSetLayout({{}, (uint32_t)bindings.size(), bindings.data()});

    vk::DescriptorPoolSize poolSize{vk::DescriptorType::eStorageBuffer, (uint32_t)buffers.size()};
    desc_pool = device.createDescriptorPool({{}, 1, 1, &poolSize});
    desc_set = device.allocateDescriptorSets({desc_pool, 1, &desc_set_layout})[0];

    std::vector<vk::WriteDescriptorSet> writes;
    std::vector<vk::DescriptorBufferInfo> dbis(buffers.size());
    for (auto i = 0u; i < buffers.size(); ++i) {
      dbis[i] = {{buffers[i], 0, VK_WHOLE_SIZE}};
      writes.push_back({desc_set, i, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &dbis[i], nullptr});
    }
    device.updateDescriptorSets(writes, {});
  }

  void initialize() {
    initialize_descriptor_set();

    std::vector<uint32_t> spv = common::read_spv("delta_step.spv");
    shader = device.createShaderModule({{}, spv.size() * 4, spv.data()});

    vk::PushConstantRange pcRange{
        vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConsts)};
    pipeline_layout = device.createPipelineLayout({{}, 1, &desc_set_layout, 1, &pcRange});

    vk::PipelineShaderStageCreateInfo shaderStage{
        {}, vk::ShaderStageFlagBits::eCompute, shader, "main"};

    pipeline = device.createComputePipeline({}, {{}, shaderStage, pipeline_layout}).value;
  }

  uint32_t run(vk::CommandPool& cmd_pool, vk::Queue& queue, uint32_t src_node, uint32_t dst_node, uint32_t delta) {
      vk::CommandBuffer cmd_buf =
          device.allocateCommandBuffers({cmd_pool, vk::CommandBufferLevel::ePrimary, 1})[0];

      const std::size_t MAX_BUCKETS = UINT32_MAX / delta - 1;

      uint32_t* gpu_changed = deltastep_buffers.changed();
      uint32_t* gpu_dist = deltastep_buffers.dist();
      auto num_nodes = (uint32_t)graph_buffers.num_nodes();

      auto changed_buffer = deltastep_buffers.buffers()[1];

      for (uint32_t bucket = 0; bucket < MAX_BUCKETS; bucket++)
      {
          *gpu_changed = 1;
          uint32_t iteration = 0;
          while (*gpu_changed > 0)
          {
              cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
              cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
              cmd_buf.bindDescriptorSets(
                  vk::PipelineBindPoint::eCompute, pipeline_layout, 0, desc_set, {});

              PushConsts pc{src_node, num_nodes, bucket, delta, iteration++, delta};

              cmd_buf.pushConstants(
                  pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);

              cmd_buf.fillBuffer(changed_buffer, 0, sizeof(uint32_t), 0);
              cmd_buf.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eComputeShader,
                vk::DependencyFlags{},
                {},
                vk::BufferMemoryBarrier{
                    vk::AccessFlagBits::eTransferWrite,
                    vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
                    VK_QUEUE_FAMILY_IGNORED,
                    VK_QUEUE_FAMILY_IGNORED,
                    changed_buffer,
                    0,
                    sizeof(uint32_t)
                },
                {});

              cmd_buf.dispatch((num_nodes + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, 1, 1);
              cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                     vk::PipelineStageFlagBits::eTransfer,
                                     vk::DependencyFlags{},
                                     {},
                                     {},
                                     {});

              cmd_buf.fillBuffer(changed_buffer, 0, sizeof(uint32_t), 0);
              cmd_buf.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eComputeShader,
                vk::DependencyFlags{},
                {},
                vk::BufferMemoryBarrier{
                    vk::AccessFlagBits::eTransferWrite,
                    vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
                    VK_QUEUE_FAMILY_IGNORED,
                    VK_QUEUE_FAMILY_IGNORED,
                    changed_buffer,
                    0,
                    sizeof(uint32_t)
                },
                {});

              cmd_buf.dispatch((num_nodes + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, 1, 1);
              cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                     vk::PipelineStageFlagBits::eTransfer,
                                     vk::DependencyFlags{},
                                     {},
                                     {},
                                     {});

              cmd_buf.fillBuffer(changed_buffer, 0, sizeof(uint32_t), 0);
              cmd_buf.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eComputeShader,
                vk::DependencyFlags{},
                {},
                vk::BufferMemoryBarrier{
                    vk::AccessFlagBits::eTransferWrite,
                    vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
                    VK_QUEUE_FAMILY_IGNORED,
                    VK_QUEUE_FAMILY_IGNORED,
                    changed_buffer,
                    0,
                    sizeof(uint32_t)
                },
                {});

              cmd_buf.dispatch((num_nodes + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, 1, 1);
              cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                     vk::PipelineStageFlagBits::eTransfer,
                                     vk::DependencyFlags{},
                                     {},
                                     {},
                                     {});

              cmd_buf.fillBuffer(changed_buffer, 0, sizeof(uint32_t), 0);
              cmd_buf.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eComputeShader,
                vk::DependencyFlags{},
                {},
                vk::BufferMemoryBarrier{
                    vk::AccessFlagBits::eTransferWrite,
                    vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
                    VK_QUEUE_FAMILY_IGNORED,
                    VK_QUEUE_FAMILY_IGNORED,
                    changed_buffer,
                    0,
                    sizeof(uint32_t)
                },
                {});

              cmd_buf.dispatch((num_nodes + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, 1, 1);
              cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                     vk::PipelineStageFlagBits::eComputeShader,
                                     vk::DependencyFlags{},
                                     {},
                                     {},
                                     {});
              cmd_buf.end();
              queue.submit(vk::SubmitInfo{0, nullptr, nullptr, 1, &cmd_buf});
              queue.waitIdle();

              //std::cout << bucket << " " << iteration << " " << *gpu_changed << std::endl;
          }

          cmd_buf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
          cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
          cmd_buf.bindDescriptorSets(
              vk::PipelineBindPoint::eCompute, pipeline_layout, 0, desc_set, {});

          PushConsts pc{src_node, num_nodes, bucket, delta, iteration++, UINT32_MAX};

          cmd_buf.pushConstants(
              pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
          cmd_buf.dispatch((num_nodes + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, 1, 1);
          cmd_buf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                 vk::PipelineStageFlagBits::eComputeShader,
                                 vk::DependencyFlags{},
                                 {},
                                 {},
                                 {});
          cmd_buf.end();
          queue.submit(vk::SubmitInfo{0, nullptr, nullptr, 1, &cmd_buf});
          queue.waitIdle();
          
          if (gpu_dist[dst_node] != UINT32_MAX) {
              // If the distance is smaller then the current bucket, we have already settled the destination
              if (gpu_dist[dst_node] < bucket * delta) {
                  break;
              }
          }
      }

      return gpu_dist[dst_node];
  }

private:
  const GraphBuffers<GraphT>& graph_buffers;
  DeltaStepBuffers& deltastep_buffers;

  vk::DescriptorSet desc_set;
  vk::DescriptorSetLayout desc_set_layout;
  vk::DescriptorPool desc_pool;
  vk::ShaderModule shader;
  vk::Pipeline pipeline;
  vk::PipelineLayout pipeline_layout;

  vk::Device& device;
};

}

#endif
