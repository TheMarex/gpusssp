#include "common/graph_metrics.hpp"
#include "common/weighted_graph.hpp"
#include "mock_graph.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstdint>

using namespace gpusssp::common;
using namespace gpusssp::test;

TEST_CASE("Delta heuristic computation", "[graph_metrics]")
{
    SECTION("Mock graph with default C=10")
    {
        auto graph = create_mock_graph();

        // Mock graph has:
        // - 6 nodes
        // - 8 edges
        // - Total weight: 1000 + 500 + 4000 + 2000 + 800 + 2500 + 500 + 4000 = 15300
        // - Average weight: 15300 / 8 = 1912.5
        // - Average degree: 8 / 6 = 1.333...
        // - Expected delta: 10 * 1912.5 / 1.333... = 14343.75

        auto delta = compute_delta_heuristic(graph);
        REQUIRE(delta == 14343);
    }

    SECTION("Mock graph with custom C=5")
    {
        auto graph = create_mock_graph();

        // With C=5, delta should be half of the default
        auto delta = compute_delta_heuristic(graph, 5.0);
        REQUIRE(delta == 7171);
    }

    SECTION("Empty graph")
    {
        WeightedGraph<uint32_t> graph;
        auto delta = compute_delta_heuristic(graph);
        REQUIRE(delta == 0);
    }
}
