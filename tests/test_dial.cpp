#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <vector>

#include "common/bucket_queue.hpp"
#include "common/constants.hpp"
#include "common/dijkstra.hpp"
#include "mock_graph.hpp"

#include "common/dial.hpp"
#include "common/edge.hpp"
#include "common/weighted_graph.hpp"

TEST_CASE("Dial computes correct shortest paths", "[dial]")
{
    auto graph = gpusssp::test::create_mock_graph();

    gpusssp::common::BucketQueue queue(graph.num_nodes(), 10000);
    gpusssp::common::CostVector<decltype(graph)> costs(graph.num_nodes(),
                                                       gpusssp::common::INF_WEIGHT);
    std::vector<bool> settled(graph.num_nodes(), false);

    for (uint32_t src_node = 0; src_node < graph.num_nodes(); ++src_node)
    {
        for (uint32_t dst_node = 0; dst_node < graph.num_nodes(); ++dst_node)
        {
            uint32_t computed_dist =
                gpusssp::common::dial(src_node, dst_node, graph, queue, costs, settled);

            INFO("Source: " << src_node << ", Destination: " << dst_node);
            auto expected = gpusssp::test::get_expected_distance(src_node, dst_node);
            REQUIRE(computed_dist == expected);
        }
    }
}

TEST_CASE("Dial handles double decrease_key on same node", "[dial]")
{
    using Edge = gpusssp::common::Edge<uint32_t, uint32_t>;

    // Triggers the stale prev pointer bug in BucketQueue::insert_entry.
    // Nodes 3, 4, 5 all land in bucket 50. Node 4 (middle of the list)
    // gets decrease_key'd twice via nodes 1 and 2. The second unlink
    // corrupts node 5's next pointer via the stale prev, severing node 3
    // from the bucket and producing a wrong shortest path to node 6.
    //
    // Shortest path to 6: 0->3(50)->6(5) = 55
    // Without the fix, node 3 is lost from bucket 50 and never popped.
    std::vector<Edge> edges = {
        Edge{0, 1, 5},
        Edge{0, 2, 15},
        Edge{0, 3, 50},
        Edge{0, 4, 50},
        Edge{0, 5, 50},
        Edge{1, 4, 30},
        Edge{2, 4, 10},
        Edge{3, 6, 5},
        Edge{4, 6, 100},
        Edge{5, 6, 100},
    };

    gpusssp::common::WeightedGraph<uint32_t> graph(7, edges);

    gpusssp::common::BucketQueue queue(graph.num_nodes(), 300);
    gpusssp::common::CostVector<decltype(graph)> costs(graph.num_nodes(),
                                                       gpusssp::common::INF_WEIGHT);
    std::vector<bool> settled(graph.num_nodes(), false);

    auto dist = gpusssp::common::dial<decltype(graph)>(0, 6, graph, queue, costs, settled);
    REQUIRE(dist == 55);
}
