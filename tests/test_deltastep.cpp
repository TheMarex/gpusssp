#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <vulkan/vulkan.hpp>
#include <limits>

#include "mock_graph.hpp"
#include "vulkan_test_fixture.hpp"

#include "gpu/graph_buffers.hpp"
#include "gpu/deltastep_buffers.hpp"
#include "gpu/deltastep.hpp"

// Placeholder test to verify Catch2 is working
TEST_CASE("Catch2 is working", "[sanity]") {
    REQUIRE(true);
    REQUIRE(1 + 1 == 2);
}

TEST_CASE("Vulkan headers are available", "[sanity]") {
    // Just verify we can use Vulkan types
    vk::ApplicationInfo appInfo;
    REQUIRE(appInfo.apiVersion == 0);
}

TEST_CASE("Mock graph is created correctly", "[mock]") {
    auto graph = gpusssp::test::create_mock_graph();
    
    REQUIRE(graph.num_nodes() == 6);
    REQUIRE(graph.num_edges() == 11);
    
    // Verify node 0 has 3 outgoing edges
    REQUIRE(graph.end(0) - graph.begin(0) == 3);
    
    // Verify first edge from node 0 goes to node 1 with weight 1000
    auto edge0 = graph.begin(0);
    REQUIRE(graph.target(edge0) == 1);
    REQUIRE(graph.weight(edge0) == 1000);
}

TEST_CASE("Expected distances are available", "[mock]") {
    auto distances = gpusssp::test::get_expected_distances(0);
    
    REQUIRE(distances.size() == 6);
    REQUIRE(distances[0] == 0);
    REQUIRE(distances[1] == 1000);
    REQUIRE(distances[3] == 500);
}

TEST_CASE("Vulkan test fixture initializes correctly", "[vulkan]") {
    gpusssp::test::VulkanTestFixture fixture;
    
    // Verify all resources are initialized
    REQUIRE(fixture.get_instance());
    REQUIRE(fixture.get_physical_device());
    REQUIRE(fixture.get_device());
    REQUIRE(fixture.get_queue());
    REQUIRE(fixture.get_command_pool());
}

TEST_CASE("DeltaStep computes correct shortest paths from node 0", "[deltastep]") {
    // Setup
    auto graph = gpusssp::test::create_mock_graph();
    gpusssp::test::VulkanTestFixture vk_fixture;
    
    auto device = vk_fixture.get_device();
    auto queue = vk_fixture.get_queue();
    auto cmd_pool = vk_fixture.get_command_pool();
    auto mem_props = vk_fixture.get_memory_properties();
    
    // Create GPU buffers
    gpusssp::gpu::GraphBuffers<gpusssp::common::WeightedGraph<uint32_t>> graph_buffers(graph, device);
    gpusssp::gpu::DeltaStepBuffers deltastep_buffers(graph.num_nodes(), device);
    
    graph_buffers.initialize(mem_props);
    deltastep_buffers.initialize(mem_props);
    
    // Create DeltaStep algorithm instance
    gpusssp::gpu::DeltaStep deltastep(graph_buffers, deltastep_buffers, device);
    deltastep.initialize();
    
    // Test parameters
    const uint32_t src_node = 0;
    const uint32_t delta = 3600;
    
    // Get expected distances
    auto expected = gpusssp::test::get_expected_distances(src_node);
    
    // Run algorithm for each destination node and verify
    for (uint32_t dst_node = 0; dst_node < graph.num_nodes(); ++dst_node) {
        uint32_t computed_dist = deltastep.run(cmd_pool, queue, src_node, dst_node, delta);
        
        INFO("Source: " << src_node << ", Destination: " << dst_node);
        REQUIRE(computed_dist == expected[dst_node]);
    }
}

TEST_CASE("DeltaStep computes correct shortest paths from node 1", "[deltastep]") {
    // Setup
    auto graph = gpusssp::test::create_mock_graph();
    gpusssp::test::VulkanTestFixture vk_fixture;
    
    auto device = vk_fixture.get_device();
    auto queue = vk_fixture.get_queue();
    auto cmd_pool = vk_fixture.get_command_pool();
    auto mem_props = vk_fixture.get_memory_properties();
    
    // Create GPU buffers
    gpusssp::gpu::GraphBuffers<gpusssp::common::WeightedGraph<uint32_t>> graph_buffers(graph, device);
    gpusssp::gpu::DeltaStepBuffers deltastep_buffers(graph.num_nodes(), device);
    
    graph_buffers.initialize(mem_props);
    deltastep_buffers.initialize(mem_props);
    
    // Create DeltaStep algorithm instance
    gpusssp::gpu::DeltaStep deltastep(graph_buffers, deltastep_buffers, device);
    deltastep.initialize();
    
    // Test parameters
    const uint32_t src_node = 1;
    const uint32_t delta = 3600;
    
    // Get expected distances
    auto expected = gpusssp::test::get_expected_distances(src_node);
    
    // Run algorithm for each destination node and verify
    for (uint32_t dst_node = 0; dst_node < graph.num_nodes(); ++dst_node) {
        uint32_t computed_dist = deltastep.run(cmd_pool, queue, src_node, dst_node, delta);
        
        INFO("Source: " << src_node << ", Destination: " << dst_node);
        REQUIRE(computed_dist == expected[dst_node]);
    }
}

TEST_CASE("DeltaStep handles source equals destination", "[deltastep]") {
    // Setup
    auto graph = gpusssp::test::create_mock_graph();
    gpusssp::test::VulkanTestFixture vk_fixture;
    
    auto device = vk_fixture.get_device();
    auto queue = vk_fixture.get_queue();
    auto cmd_pool = vk_fixture.get_command_pool();
    auto mem_props = vk_fixture.get_memory_properties();
    
    // Create GPU buffers
    gpusssp::gpu::GraphBuffers<gpusssp::common::WeightedGraph<uint32_t>> graph_buffers(graph, device);
    gpusssp::gpu::DeltaStepBuffers deltastep_buffers(graph.num_nodes(), device);
    
    graph_buffers.initialize(mem_props);
    deltastep_buffers.initialize(mem_props);
    
    // Create DeltaStep algorithm instance
    gpusssp::gpu::DeltaStep deltastep(graph_buffers, deltastep_buffers, device);
    deltastep.initialize();
    
    const uint32_t delta = 3600;
    
    // Test that distance from any node to itself is 0
    for (uint32_t node = 0; node < graph.num_nodes(); ++node) {
        uint32_t computed_dist = deltastep.run(cmd_pool, queue, node, node, delta);
        
        INFO("Node: " << node);
        REQUIRE(computed_dist == 0);
    }
}

TEST_CASE("DeltaStep handles unreachable nodes", "[deltastep]") {
    // Setup
    auto graph = gpusssp::test::create_mock_graph();
    gpusssp::test::VulkanTestFixture vk_fixture;
    
    auto device = vk_fixture.get_device();
    auto queue = vk_fixture.get_queue();
    auto cmd_pool = vk_fixture.get_command_pool();
    auto mem_props = vk_fixture.get_memory_properties();
    
    // Create GPU buffers
    gpusssp::gpu::GraphBuffers<gpusssp::common::WeightedGraph<uint32_t>> graph_buffers(graph, device);
    gpusssp::gpu::DeltaStepBuffers deltastep_buffers(graph.num_nodes(), device);
    
    graph_buffers.initialize(mem_props);
    deltastep_buffers.initialize(mem_props);
    
    // Create DeltaStep algorithm instance
    gpusssp::gpu::DeltaStep deltastep(graph_buffers, deltastep_buffers, device);
    deltastep.initialize();
    
    const uint32_t delta = 3600;
    
    // Node 5 has no outgoing edges, so all other nodes are unreachable from it
    const uint32_t src_node = 5;
    
    for (uint32_t dst_node = 0; dst_node < graph.num_nodes(); ++dst_node) {
        if (dst_node == src_node) continue;  // Skip self
        
        uint32_t computed_dist = deltastep.run(cmd_pool, queue, src_node, dst_node, delta);
        
        INFO("Source: " << src_node << ", Destination: " << dst_node);
        REQUIRE(computed_dist == UINT32_MAX);
    }
}
