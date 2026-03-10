#ifndef GPUSSSP_COMMON_GRAPH_METRICS_HPP
#define GPUSSSP_COMMON_GRAPH_METRICS_HPP

#include "common/weighted_graph.hpp"
#include <algorithm>
#include <numeric>

namespace gpusssp::common
{

inline std::size_t max_degree(const AdjGraph &graph)
{
    std::size_t max_deg = 0;
    for (AdjGraph::node_id_t node = 0; node < graph.num_nodes(); ++node)
    {
        auto deg = static_cast<std::size_t>(graph.end(node) - graph.begin(node));
        max_deg = std::max(max_deg, deg);
    }
    return max_deg;
}

template <typename WeightT>
uint32_t compute_delta_heuristic(const WeightedGraph<WeightT> &graph, double c = 10.0)
{
    if (graph.num_edges() == 0 || graph.num_nodes() == 0)
    {
        return 0.0;
    }

    double total_weight =
        std::accumulate(graph.weights.begin(),
                        graph.weights.end(),
                        0.0,
                        [](double sum, WeightT w) { return sum + static_cast<double>(w); });
    double avg_weight = total_weight / graph.num_edges();

    double avg_degree = static_cast<double>(graph.num_edges()) / graph.num_nodes();

    return static_cast<uint32_t>(c * avg_weight / avg_degree);
}

} // namespace gpusssp::common

#endif
