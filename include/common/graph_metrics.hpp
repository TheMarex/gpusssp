#ifndef GPUSSSP_COMMON_GRAPH_METRICS_HPP
#define GPUSSSP_COMMON_GRAPH_METRICS_HPP

#include "common/weighted_graph.hpp"
#include <numeric>

namespace gpusssp
{
namespace common
{

template <typename WeightT>
uint32_t compute_delta_heuristic(const WeightedGraph<WeightT> &graph, double C = 10.0)
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

    return static_cast<uint32_t>(C * avg_weight / avg_degree);
}

} // namespace common
} // namespace gpusssp

#endif
