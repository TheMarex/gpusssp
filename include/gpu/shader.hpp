#ifndef GPUSSSP_GPU_SHADER_HPP
#define GPUSSSP_GPU_SHADER_HPP

#include <type_traits>
#include <vector>
#include <vulkan/vulkan.hpp>

#include "common/shader.hpp"

namespace gpusssp::gpu
{

namespace detail
{
struct DescriptorSetBundle
{
    std::vector<vk::DescriptorSet> descriptor_sets;
    vk::DescriptorSetLayout layout;
    vk::DescriptorPool pool;
};

inline DescriptorSetBundle
create_descriptor_sets(vk::Device device, const std::vector<std::vector<vk::Buffer>> &buffer_sets)
{
    if (buffer_sets.empty())
    {
        throw std::runtime_error("buffer_sets cannot be empty");
    }

    const auto num_descriptor_sets = static_cast<uint32_t>(buffer_sets.size());
    const auto num_bindings = static_cast<uint32_t>(buffer_sets[0].size());

    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    bindings.reserve(num_bindings);
    for (uint32_t i = 0; i < num_bindings; ++i)
    {
        bindings.emplace_back(
            i, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);
    }

    vk::DescriptorSetLayout layout =
        device.createDescriptorSetLayout({{}, num_bindings, bindings.data()});

    vk::DescriptorPoolSize pool_size{vk::DescriptorType::eStorageBuffer,
                                     num_bindings * num_descriptor_sets};
    vk::DescriptorPool pool = device.createDescriptorPool({{}, num_descriptor_sets, 1, &pool_size});

    std::vector<vk::DescriptorSetLayout> layouts(num_descriptor_sets, layout);
    std::vector<vk::DescriptorSet> descriptor_sets =
        device.allocateDescriptorSets({pool, num_descriptor_sets, layouts.data()});

    for (uint32_t set_idx = 0; set_idx < num_descriptor_sets; ++set_idx)
    {
        const auto &buffers = buffer_sets[set_idx];
        if (buffers.size() != num_bindings)
        {
            throw std::runtime_error("All descriptor sets must have the same number of buffers");
        }

        std::vector<vk::DescriptorBufferInfo> buffer_infos;
        buffer_infos.reserve(num_bindings);
        for (const auto &buffer : buffers)
        {
            buffer_infos.emplace_back(buffer, 0, VK_WHOLE_SIZE);
        }

        std::vector<vk::WriteDescriptorSet> writes;
        writes.reserve(num_bindings);
        for (uint32_t binding_idx = 0; binding_idx < num_bindings; ++binding_idx)
        {
            writes.emplace_back(descriptor_sets[set_idx],
                                binding_idx,
                                0,
                                1,
                                vk::DescriptorType::eStorageBuffer,
                                nullptr,
                                &buffer_infos[binding_idx],
                                nullptr);
        }

        device.updateDescriptorSets(writes, {});
    }

    return {.descriptor_sets = descriptor_sets, .layout = layout, .pool = pool};
}
} // namespace detail

inline vk::ShaderModule create_shader_module(vk::Device device, const std::string &shader_path)
{
    std::vector<uint32_t> spv = common::read_spv(shader_path);
    return device.createShaderModule({{}, spv.size() * 4, spv.data()});
}

struct ComputePipeline
{
    vk::ShaderModule shader;
    vk::Pipeline pipeline;
    vk::PipelineLayout layout;
    std::vector<vk::DescriptorSet> descriptor_sets;
    vk::DescriptorSetLayout descriptor_set_layout;
    vk::DescriptorPool descriptor_pool;

    void destroy(vk::Device device)
    {
        device.destroyPipeline(pipeline);
        device.destroyPipelineLayout(layout);
        device.destroyDescriptorSetLayout(descriptor_set_layout);
        device.destroyDescriptorPool(descriptor_pool);
        device.destroyShaderModule(shader);
    }
};

template <typename PushConstantsT = void>
inline ComputePipeline
create_compute_pipeline(vk::Device device,
                        const std::string &shader_path,
                        const std::vector<std::vector<vk::Buffer>> &buffer_sets,
                        const std::vector<uint32_t> &specialization_constants = {})
{
    auto [descriptor_sets, descriptor_set_layout, descriptor_pool] =
        detail::create_descriptor_sets(device, buffer_sets);

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
            spec_entries.emplace_back(
                i, static_cast<uint32_t>(i * sizeof(uint32_t)), sizeof(uint32_t));
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

    return {.shader = shader,
            .pipeline = pipeline,
            .layout = pipeline_layout,
            .descriptor_sets = std::move(descriptor_sets),
            .descriptor_set_layout = descriptor_set_layout,
            .descriptor_pool = descriptor_pool};
}

} // namespace gpusssp::gpu

#endif
