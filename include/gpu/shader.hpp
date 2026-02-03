#ifndef GPUSSSP_GPU_SHADER_HPP
#define GPUSSSP_GPU_SHADER_HPP

#include <type_traits>
#include <vector>
#include <vulkan/vulkan.hpp>

#include "common/shader.hpp"

namespace gpusssp::gpu
{

struct ComputePipeline
{
    vk::ShaderModule shader;
    vk::Pipeline pipeline;
    vk::PipelineLayout layout;
};

template <typename PushConstantsT = void>
inline ComputePipeline create_compute_pipeline(vk::Device &device,
                                                const std::string &shader_path,
                                                vk::DescriptorSetLayout descriptor_set_layout,
                                                const std::vector<uint32_t> &specialization_constants = {})
{
    std::vector<uint32_t> spv = common::read_spv(shader_path);
    vk::ShaderModule shader = device.createShaderModule({{}, spv.size() * 4, spv.data()});

    vk::PushConstantRange push_constant_range{};
    uint32_t num_push_constant_ranges = 0;
    if constexpr (!std::is_same_v<PushConstantsT, void>)
    {
        push_constant_range = {vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstantsT)};
        num_push_constant_ranges = 1;
    }

    vk::PipelineLayout pipeline_layout =
        device.createPipelineLayout({{},
                                     1,
                                     &descriptor_set_layout,
                                     num_push_constant_ranges,
                                     num_push_constant_ranges ? &push_constant_range : nullptr});

    vk::SpecializationInfo spec_info{};
    std::vector<vk::SpecializationMapEntry> spec_entries;
    if (!specialization_constants.empty())
    {
        spec_entries.reserve(specialization_constants.size());
        for (uint32_t i = 0; i < specialization_constants.size(); ++i)
        {
            spec_entries.push_back(
                {i, static_cast<uint32_t>(i * sizeof(uint32_t)), sizeof(uint32_t)});
        }
        spec_info = {static_cast<uint32_t>(spec_entries.size()),
                     spec_entries.data(),
                     specialization_constants.size() * sizeof(uint32_t),
                     specialization_constants.data()};
    }

    vk::PipelineShaderStageCreateInfo shader_stage{{},
                                                   vk::ShaderStageFlagBits::eCompute,
                                                   shader,
                                                   "main",
                                                   specialization_constants.empty() ? nullptr
                                                                                    : &spec_info};

    vk::Pipeline pipeline =
        device.createComputePipeline({}, {{}, shader_stage, pipeline_layout}).value;

    return {shader, pipeline, pipeline_layout};
}

} // namespace gpusssp::gpu

#endif
