#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vulkan/vulkan.hpp>

#include "common/dijkstra.hpp"
#include "common/files.hpp"
#include "common/id_queue.hpp"
#include "common/weighted_graph.hpp"
#include "gpu/statistics.hpp"
#include "mock_graph.hpp"
#include "vulkan_test_fixture.hpp"

#include "gpu/graph_buffers.hpp"
#include "gpu/nearfar.hpp"
#include "gpu/nearfar_buffers.hpp"

TEST_CASE("NearFar computes correct shortest paths", "[nearfar]")
{
    auto graph = gpusssp::test::create_mock_graph();
    gpusssp::test::VulkanTestFixture vk_fixture;

    auto device = vk_fixture.get_device();
    auto queue = vk_fixture.get_queue();
    auto cmd_pool = vk_fixture.get_command_pool();
    auto mem_props = vk_fixture.get_memory_properties();

    gpusssp::gpu::GraphBuffers<gpusssp::common::WeightedGraph<uint32_t>> graph_buffers(
        graph, device, mem_props, cmd_pool, queue);
    gpusssp::gpu::NearFarBuffers nearfar_buffers(graph.num_nodes(), device, mem_props);
    gpusssp::gpu::Statistics statistics(device, mem_props);
    const uint32_t delta = 3600;
    gpusssp::gpu::NearFar nearfar(graph_buffers, nearfar_buffers, device, statistics, delta);
    nearfar.initialize(cmd_pool);

    for (uint32_t src_node = 0; src_node < graph.num_nodes(); ++src_node)
    {
        for (uint32_t dst_node = 0; dst_node < graph.num_nodes(); ++dst_node)
        {
            uint32_t computed_dist = nearfar.run(cmd_pool, queue, src_node, dst_node);

            INFO("Source: " << src_node << ", Destination: " << dst_node);
            auto expected = gpusssp::test::get_expected_distances(src_node, dst_node);
            REQUIRE(computed_dist == expected);
        }
    }
}

TEST_CASE("NearFar computes correct shortest paths with batch size", "[nearfar]")
{
    auto graph = gpusssp::test::create_mock_graph();
    gpusssp::test::VulkanTestFixture vk_fixture;

    auto device = vk_fixture.get_device();
    auto queue = vk_fixture.get_queue();
    auto cmd_pool = vk_fixture.get_command_pool();
    auto mem_props = vk_fixture.get_memory_properties();

    gpusssp::gpu::GraphBuffers<gpusssp::common::WeightedGraph<uint32_t>> graph_buffers(
        graph, device, mem_props, cmd_pool, queue);
    gpusssp::gpu::NearFarBuffers nearfar_buffers(graph.num_nodes(), device, mem_props);
    gpusssp::gpu::Statistics statistics(device, mem_props);
    const uint32_t delta = 3600;
    gpusssp::gpu::NearFar nearfar(graph_buffers, nearfar_buffers, device, statistics, delta, 1);
    nearfar.initialize(cmd_pool);

    for (uint32_t src_node = 0; src_node < graph.num_nodes(); ++src_node)
    {
        for (uint32_t dst_node = 0; dst_node < graph.num_nodes(); ++dst_node)
        {
            uint32_t computed_dist = nearfar.run(cmd_pool, queue, src_node, dst_node);

            INFO("Source: " << src_node << ", Destination: " << dst_node);
            auto expected = gpusssp::test::get_expected_distances(src_node, dst_node);
            REQUIRE(computed_dist == expected);
        }
    }
}

#ifdef ENABLE_STATISTICS
TEST_CASE("NearFar statistics are collected", "[nearfar][statistics]")
{
    auto graph = gpusssp::test::create_mock_graph();
    gpusssp::test::VulkanTestFixture vk_fixture;

    auto device = vk_fixture.get_device();
    auto queue = vk_fixture.get_queue();
    auto cmd_pool = vk_fixture.get_command_pool();
    auto mem_props = vk_fixture.get_memory_properties();

    gpusssp::gpu::GraphBuffers<gpusssp::common::WeightedGraph<uint32_t>> graph_buffers(
        graph, device, mem_props, cmd_pool, queue);
    gpusssp::gpu::NearFarBuffers nearfar_buffers(graph.num_nodes(), device, mem_props);

    gpusssp::gpu::Statistics statistics(device, mem_props);
    const uint32_t delta = 3600;
    gpusssp::gpu::NearFar nearfar(graph_buffers, nearfar_buffers, device, statistics, delta);
    nearfar.initialize(cmd_pool);

    const uint32_t src_node = 0;
    const uint32_t dst_node = 3;

    statistics.reset();
    uint32_t computed_dist = nearfar.run(cmd_pool, queue, src_node, dst_node);

    auto summary = statistics.summary();
    INFO("Statistics summary:\n" << summary);

    REQUIRE(computed_dist == gpusssp::test::get_expected_distances(src_node, dst_node));
    REQUIRE_FALSE(summary.empty());
}
#endif

TEST_CASE("NearFar regression test - minimal subgraph bug", "[nearfar][regression]")
{
    auto graph = gpusssp::common::files::read_weighted_graph<uint32_t>("../cache/test");
    gpusssp::test::VulkanTestFixture vk_fixture;

    auto device = vk_fixture.get_device();
    auto queue = vk_fixture.get_queue();
    auto cmd_pool = vk_fixture.get_command_pool();
    auto mem_props = vk_fixture.get_memory_properties();

    gpusssp::gpu::GraphBuffers<gpusssp::common::WeightedGraph<uint32_t>> graph_buffers(
        graph, device, mem_props, cmd_pool, queue);
    gpusssp::gpu::NearFarBuffers nearfar_buffers(graph.num_nodes(), device, mem_props);
    gpusssp::gpu::Statistics statistics(device, mem_props);

    const uint32_t source = 1148;
    const uint32_t target = 193;
    const uint32_t delta = 7200;

    gpusssp::common::MinIDQueue min_queue(graph.num_nodes());
    gpusssp::common::CostVector<gpusssp::common::WeightedGraph<uint32_t>> costs(
        graph.num_nodes(), gpusssp::common::INF_WEIGHT);
    std::vector<bool> settled(graph.num_nodes(), false);

    auto expected_dist =
        gpusssp::common::dijkstra(source, target, graph, min_queue, costs, settled);

    gpusssp::gpu::NearFar nearfar(graph_buffers, nearfar_buffers, device, statistics, delta);
    nearfar.initialize(cmd_pool);

    uint32_t computed_dist = nearfar.run(cmd_pool, queue, source, target);

    INFO("Source: " << source << ", Target: " << target << ", Delta: " << delta);
    INFO("Expected (Dijkstra): " << expected_dist << ", Computed (NearFar): " << computed_dist);
    REQUIRE(computed_dist == expected_dist);
}
