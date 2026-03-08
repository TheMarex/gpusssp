#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vulkan/vulkan.hpp>

#include "common/weighted_graph.hpp"
#include "mock_graph.hpp"
#include "vulkan_test_fixture.hpp"

#include "gpu/deltastep.hpp"
#include "gpu/deltastep_buffers.hpp"
#include "gpu/graph_buffers.hpp"
#include "gpu/statistics.hpp"

TEST_CASE("DeltaStep computes correct shortest paths", "[deltastep]")
{
    // Setup
    auto graph = gpusssp::test::create_mock_graph();
    gpusssp::test::VulkanTestFixture vk_fixture;

    auto device = vk_fixture.get_device();
    auto queue = vk_fixture.get_queue();
    auto cmd_pool = vk_fixture.get_command_pool();
    auto mem_props = vk_fixture.get_memory_properties();

    gpusssp::gpu::GraphBuffers<gpusssp::common::WeightedGraph<uint32_t>> graph_buffers(
        graph, device, mem_props, cmd_pool, queue);
    gpusssp::gpu::DeltaStepBuffers deltastep_buffers(graph.num_nodes(), device, mem_props);
    gpusssp::gpu::Statistics gpu_statistics(device, mem_props);

    const uint32_t delta = 3600;
    gpusssp::gpu::DeltaStep deltastep(
        graph_buffers, deltastep_buffers, device, gpu_statistics, delta);
    deltastep.initialize(cmd_pool);

    // Test parameters

    // Get expected distances

    for (uint32_t src_node = 0; src_node < graph.num_nodes(); ++src_node)
    {
        for (uint32_t dst_node = 0; dst_node < graph.num_nodes(); ++dst_node)
        {
            uint32_t computed_dist = deltastep.run(cmd_pool, queue, src_node, dst_node);

            INFO("Source: " << src_node << ", Destination: " << dst_node);
            auto expected = gpusssp::test::get_expected_distances(src_node, dst_node);
            REQUIRE(computed_dist == expected);
        }
    }
}
