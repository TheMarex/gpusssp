#ifndef GPUSSSP_ADJ_GRAPH_HPP
#define GPUSSSP_ADJ_GRAPH_HPP

#include "common/constants.hpp"
#include "common/edge.hpp"
#include "common/irange.hpp"

#include <cassert>
#include <cstdint>
#include <tuple>
#include <vector>

namespace gpusssp::common
{

class AdjGraph
{
  public:
    using node_id_t = std::uint32_t;
    using edge_id_t = std::uint32_t;
    using edge_t = Edge<node_id_t, void>;
    using edge_range_t = decltype(irange<edge_id_t>(0, 1));
    using node_range_t = decltype(irange<node_id_t>(0, 1));

    AdjGraph() = default;

    AdjGraph(std::vector<edge_id_t> first_edges, std::vector<node_id_t> targets)
        : first_edges(std::move(first_edges)), targets(std::move(targets))
    {
        assert(first_edges.size() > 0);
    }

    template <typename EdgeT>
    AdjGraph(std::size_t num_nodes_, const std::vector<EdgeT> &sorted_edges) // NOLINT
    {
        assert(std::is_sorted(
            sorted_edges.begin(),
            sorted_edges.end(),
            [](const auto &lhs, const auto &rhs)
            { return std::tie(lhs.start, lhs.target) < std::tie(rhs.start, rhs.target); }));
        auto last_id = INVALID_ID;
        for (const auto &edge : sorted_edges)
        {
            // we need this to fill any gaps
            while (first_edges.size() < edge.start)
            {
                first_edges.push_back(targets.size());
            }

            if (last_id != edge.start)
            {
                first_edges.push_back(targets.size());
                last_id = edge.start;
            }
            targets.push_back(edge.target);
        }
        // fill up graps at the end
        while (first_edges.size() < num_nodes_ + 1)
        {
            first_edges.push_back(targets.size());
        }

        assert(first_edges.size() == num_nodes_ + 1);
        assert(targets.size() == sorted_edges.size());
    }

    [[nodiscard]] std::size_t num_nodes() const { return first_edges.size() - 1; }

    [[nodiscard]] std::size_t num_edges() const { return targets.size(); }

    [[nodiscard]] node_range_t nodes() const { return irange<node_id_t>(0, num_nodes()); }

    [[nodiscard]] edge_range_t edges(node_id_t node) const
    {
        return irange<edge_id_t>(begin(node), end(node));
    }

    [[nodiscard]] edge_id_t begin(node_id_t node) const { return first_edges[node]; }

    [[nodiscard]] edge_id_t end(node_id_t node) const { return first_edges[node + 1]; }

    [[nodiscard]] node_id_t target(edge_id_t edge) const { return targets[edge]; }

    [[nodiscard]] edge_id_t edge(node_id_t start_node, node_id_t target_node) const
    {
        for (auto edge = begin(start_node); edge < end(start_node); ++edge)
        {
            if (target(edge) == target_node)
                return edge;
        }

        return INVALID_ID;
    }

    [[nodiscard]] std::vector<edge_t> edges() const
    {
        std::vector<edge_t> edges;
        edges.reserve(num_edges());

        for (node_id_t node = 0; node < num_nodes(); ++node)
        {
            for (auto edge = begin(node); edge < end(node); ++edge)
            {
                edges.emplace_back(node, target(edge));
            }
        }

        return edges;
    }

    static std::tuple<std::vector<edge_id_t>, std::vector<node_id_t>> unwrap(AdjGraph graph)
    {
        return std::tie(graph.first_edges, graph.targets);
    }

    std::vector<edge_id_t> first_edges;
    std::vector<node_id_t> targets;
};
} // namespace gpusssp::common

#endif
