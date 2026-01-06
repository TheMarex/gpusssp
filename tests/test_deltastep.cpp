#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <vulkan/vulkan.hpp>
#include <limits>

#include "mock_graph.hpp"

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
