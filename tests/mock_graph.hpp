#ifndef GPUSSSP_TESTS_MOCK_GRAPH_HPP
#define GPUSSSP_TESTS_MOCK_GRAPH_HPP

#include "common/constants.hpp"
#include "common/edge.hpp"
#include "common/weighted_graph.hpp"
#include <cstdint>
#include <vector>

namespace gpusssp::test
{

//
// Graph topology (6 nodes, 8 edges):
//
//           1000    2000
// |----- 0 -----> 1 -----> 2
// |      |        |        |
// |   500|     800|        | 2500
// |      v        v        v
// |      3 -----> 4 -----> 5
// |         500   ^  4000
// |---------------|
//     4000
//

inline common::WeightedGraph<uint32_t> create_mock_graph()
{
    using Edge = common::Edge<uint32_t, uint32_t>;

    const size_t num_nodes = 6;

    std::vector<Edge> edges = {
        Edge{0, 1, 1000},
        Edge{0, 3, 500},
        Edge{0, 4, 4000},
        Edge{1, 2, 2000},
        Edge{1, 4, 800},
        Edge{2, 5, 2500},
        Edge{3, 4, 500},
        Edge{4, 5, 4000},
        // Node 5 has no outgoing edges
    };

    return common::WeightedGraph<uint32_t>(num_nodes, edges);
}

inline uint32_t get_expected_distances(uint32_t src_node, uint32_t dst_node)
{
    using common::INF_WEIGHT;

    static constexpr std::array<uint32_t, 36> SHORTEST_DISTANCES_MOCK = {
        // 0        1           2           3           4           5
        0,          1000,       3000,       500,        1000,       5000,
        INF_WEIGHT, 0,          2000,       INF_WEIGHT, 800,        4500,
        INF_WEIGHT, INF_WEIGHT, 0,          INF_WEIGHT, INF_WEIGHT, 2500,
        INF_WEIGHT, INF_WEIGHT, INF_WEIGHT, 0,          500,        4500,
        INF_WEIGHT, INF_WEIGHT, INF_WEIGHT, INF_WEIGHT, 0,          4000,
        INF_WEIGHT, INF_WEIGHT, INF_WEIGHT, INF_WEIGHT, INF_WEIGHT, 0};
    return SHORTEST_DISTANCES_MOCK[(src_node * 6) + dst_node];
}

} // namespace gpusssp::test

#endif // GPUSSSP_TESTS_MOCK_GRAPH_HPP
