#ifndef GPUSSSP_COMMON_GRAPH_METRICS_HPP
#define GPUSSSP_COMMON_GRAPH_METRICS_HPP

#include "common/constants.hpp"
#include "common/coordinate.hpp"
#include "common/weighted_graph.hpp"
#include <numeric>

namespace gpusssp::common
{

inline uint64_t estimate_max_path_weight(const std::vector<Coordinate> &coordinates,
                                         double detour_factor = 1.7,
                                         double min_speed_kmh = 30.0)
{
    auto bbox = bounds(coordinates);
    auto diagonal_meters = haversine_distance(bbox.south_east, bbox.north_west);
    auto min_speed_ms = min_speed_kmh / 3.6;
    auto max_travel_seconds = detour_factor * diagonal_meters / min_speed_ms;
    return static_cast<uint64_t>(max_travel_seconds * FIXED_POINT_RESOLUTION);
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
