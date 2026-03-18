#ifndef GPUSSSP_TESTS_GRID_GRAPH_HPP
#define GPUSSSP_TESTS_GRID_GRAPH_HPP

#include "common/edge.hpp"
#include "common/weighted_graph.hpp"
#include <cstdint>
#include <vector>

namespace gpusssp::test
{

inline common::WeightedGraph<uint32_t> create_grid_graph(size_t width, size_t height)
{
    using Edge = common::Edge<uint32_t, uint32_t>;

    const size_t num_nodes = width * height;
    std::vector<Edge> edges;

    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            const auto node_id = static_cast<uint32_t>((y * width) + x);

            if (x > 0)
            {
                const auto neighbor_id = static_cast<uint32_t>((y * width) + (x - 1));
                edges.emplace_back(node_id, neighbor_id, 100);
            }

            if (x + 1 < width)
            {
                const auto neighbor_id = static_cast<uint32_t>((y * width) + (x + 1));
                edges.emplace_back(node_id, neighbor_id, 100);
            }

            if (y > 0)
            {
                const auto neighbor_id = static_cast<uint32_t>(((y - 1) * width) + x);
                edges.emplace_back(node_id, neighbor_id, 100);
            }

            if (y + 1 < height)
            {
                const auto neighbor_id = static_cast<uint32_t>(((y + 1) * width) + x);
                edges.emplace_back(node_id, neighbor_id, 100);
            }
        }
    }

    std::sort(edges.begin(), edges.end());

    return common::WeightedGraph<uint32_t>(num_nodes, edges);
}

inline uint32_t get_expected_distance(uint32_t src_node, uint32_t dst_node, size_t width)
{
    const size_t src_x = src_node % width;
    const size_t src_y = src_node / width;
    const size_t dst_x = dst_node % width;
    const size_t dst_y = dst_node / width;

    const size_t dx = (src_x > dst_x) ? (src_x - dst_x) : (dst_x - src_x);
    const size_t dy = (src_y > dst_y) ? (src_y - dst_y) : (dst_y - src_y);

    return static_cast<uint32_t>((dx + dy) * 100);
}

} // namespace gpusssp::test

#endif // GPUSSSP_TESTS_GRID_GRAPH_HPP
