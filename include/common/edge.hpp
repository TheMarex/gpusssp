#ifndef GPUSSSP_COMMON_EDGE_HPP
#define GPUSSSP_COMMON_EDGE_HPP

#include "common/constants.hpp"

#include <tuple>

namespace gpusssp::common
{
template <typename NodeT, typename WeightT> struct Edge
{
    using node_t = NodeT;
    using weight_t = WeightT;

    Edge() : start(INVALID_ID), target(INVALID_ID), weight{} {}

    Edge(NodeT start, NodeT target, WeightT weight) : start(start), target(target), weight(weight)
    {
    }

    bool operator<(const Edge &rhs) const
    {
        return std::tie(start, target, weight) < std::tie(rhs.start, rhs.target, rhs.weight);
    }

    bool operator>(const Edge &rhs) const
    {
        return std::tie(start, target, weight) > std::tie(rhs.start, rhs.target, rhs.weight);
    }

    bool operator>=(const Edge &rhs) const
    {
        return std::tie(start, target, weight) >= std::tie(rhs.start, rhs.target, rhs.weight);
    }

    bool operator<=(const Edge &rhs) const
    {
        return std::tie(start, target, weight) <= std::tie(rhs.start, rhs.target, rhs.weight);
    }

    bool operator==(const Edge &rhs) const
    {
        return std::tie(start, target, weight) == std::tie(rhs.start, rhs.target, rhs.weight);
    }

    bool operator!=(const Edge &rhs) const { return !operator==(rhs); }

    NodeT start;
    NodeT target;
    WeightT weight;

    static_assert(std::totally_ordered<Edge<NodeT, WeightT>>, "Needs to be totally_ordered");
};

template <typename NodeT> struct Edge<NodeT, void>
{
    using node_t = NodeT;

    Edge() : start(INVALID_ID), target(INVALID_ID) {}

    Edge(NodeT start, NodeT target) : start(start), target(target) {}

    bool operator<(const Edge &rhs) const
    {
        return std::tie(start, target) < std::tie(rhs.start, rhs.target);
    }

    bool operator>(const Edge &rhs) const
    {
        return std::tie(start, target) > std::tie(rhs.start, rhs.target);
    }

    bool operator>=(const Edge &rhs) const
    {
        return std::tie(start, target) >= std::tie(rhs.start, rhs.target);
    }

    bool operator<=(const Edge &rhs) const
    {
        return std::tie(start, target) <= std::tie(rhs.start, rhs.target);
    }

    bool operator==(const Edge &rhs) const
    {
        return std::tie(start, target) == std::tie(rhs.start, rhs.target);
    }

    bool operator!=(const Edge &rhs) const { return !operator==(rhs); }

    NodeT start;
    NodeT target;

    static_assert(std::totally_ordered<Edge<NodeT, void>>, "Needs to be totally_ordered");
};
} // namespace gpusssp::common

#endif
