#ifndef GPUSSSP_TESTS_MOCK_GRAPH_HPP
#define GPUSSSP_TESTS_MOCK_GRAPH_HPP

#include "common/weighted_graph.hpp"
#include "common/edge.hpp"
#include <vector>
#include <cstdint>

namespace gpusssp::test {

/**
 * Creates a small mock graph for testing purposes.
 * 
 * Graph topology (6 nodes, 11 edges):
 * 
 *        1000      2000
 *     0 -----> 1 -----> 2
 *     |        |        |
 *  500|     800|        | 1500
 *     v        v        v
 *     3 -----> 4 -----> 5
 *        1200      3000
 * 
 * Additional edges for more interesting paths:
 *   0 -> 4: 5000 (heavy edge, weight > 3600)
 *   1 -> 5: 4500 (heavy edge)
 *   2 -> 3: 6000 (heavy edge)
 *   3 -> 2: 4000 (heavy edge)
 * 
 * Expected shortest paths from node 0 (with delta=3600):
 *   0 -> 0: 0
 *   0 -> 1: 1000
 *   0 -> 2: 3000
 *   0 -> 3: 500
 *   0 -> 4: 1700  (via 0->3->4)
 *   0 -> 5: 4700  (via 0->3->4->5)
 * 
 * Expected shortest paths from node 1:
 *   1 -> 0: UINT32_MAX (unreachable)
 *   1 -> 1: 0
 *   1 -> 2: 2000
 *   1 -> 3: 800
 *   1 -> 4: 800
 *   1 -> 5: 3800
 * 
 * Expected shortest paths from node 2:
 *   2 -> 0: UINT32_MAX (unreachable)
 *   2 -> 1: UINT32_MAX (unreachable)
 *   2 -> 2: 0
 *   2 -> 3: 1500
 *   2 -> 4: 2700
 *   2 -> 5: 1500
 */
inline common::WeightedGraph<uint32_t> create_mock_graph() {
    using Edge = common::Edge<uint32_t, uint32_t>;
    
    const size_t num_nodes = 6;
    
    // Define edges (must be sorted by (start, target))
    std::vector<Edge> edges = {
        // From node 0
        Edge{0, 1, 1000},
        Edge{0, 3, 500},
        Edge{0, 4, 5000},  // heavy edge
        
        // From node 1
        Edge{1, 2, 2000},
        Edge{1, 4, 800},
        Edge{1, 5, 4500},  // heavy edge
        
        // From node 2
        Edge{2, 3, 6000},  // heavy edge
        Edge{2, 5, 1500},
        
        // From node 3
        Edge{3, 2, 4000},  // heavy edge
        Edge{3, 4, 1200},
        
        // From node 4
        Edge{4, 5, 3000},
        
        // Node 5 has no outgoing edges
    };
    
    return common::WeightedGraph<uint32_t>(num_nodes, edges);
}

/**
 * Returns expected shortest path distances from a given source node.
 * Returns a vector of size 6 with distances to each node.
 * UINT32_MAX indicates unreachable nodes.
 */
inline std::vector<uint32_t> get_expected_distances(uint32_t src_node) {
    switch (src_node) {
        case 0:
            return {0, 1000, 3000, 500, 1700, 4700};
        case 1:
            return {UINT32_MAX, 0, 2000, 800, 800, 3800};
        case 2:
            return {UINT32_MAX, UINT32_MAX, 0, 1500, 2700, 1500};
        case 3:
            return {UINT32_MAX, UINT32_MAX, 4000, 0, 1200, 4200};
        case 4:
            return {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, 0, 3000};
        case 5:
            return {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, 0};
        default:
            return std::vector<uint32_t>(6, UINT32_MAX);
    }
}

} // namespace gpusssp::test

#endif // GPUSSSP_TESTS_MOCK_GRAPH_HPP
