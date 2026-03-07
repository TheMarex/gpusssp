#ifndef GPUSSSP_COMMON_DIAL_HPP
#define GPUSSSP_COMMON_DIAL_HPP

#include "common/bucket_queue.hpp"
#include "common/dijkstra.hpp"

namespace gpusssp::common
{

template <typename GraphT>
auto dial(typename GraphT::node_id_t source,
          typename GraphT::node_id_t target,
          const GraphT &graph,
          BucketQueue &queue,
          CostVector<GraphT> &costs,
          std::vector<bool> &settled)
{
    queue.clear();
    costs.clear();
    std::fill(settled.begin(), settled.end(), false); // NOLINT
    costs[source] = 0;
    queue.push({source, 0});

    while (!queue.empty())
    {
        auto id = queue.peek().id;
        settled[id] = true;

        detail::route_step(queue, costs, graph);

        if (id == target)
        {
            break;
        }
    }

    return costs[target];
}
} // namespace gpusssp::common

#endif
