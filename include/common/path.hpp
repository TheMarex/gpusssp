#ifndef GPUSSSP_COMMON_PATH_HPP
#define GPUSSSP_COMMON_PATH_HPP

#include "common/dijkstra.hpp"

#include <vector>

namespace gpusssp::common
{

namespace detail
{
template <typename GraphT, typename OutIter>
void backtrack_path(const typename GraphT::node_id_t from,
                    const ParentVector<GraphT> &parents,
                    OutIter iter)
{
    auto node = from;
    do
    {
        *iter++ = node;
        node = parents[node];
    } while (node != INVALID_ID);
}

} // namespace detail

template <typename GraphT>
auto get_path(const typename GraphT::node_id_t start,
              const typename GraphT::node_id_t middle,
              const typename GraphT::node_id_t target,
              const ParentVector<GraphT> &forward_parents,
              const ParentVector<GraphT> &reverse_parents)
{
    std::vector<typename GraphT::node_id_t> path;
    detail::backtrack_path<GraphT>(middle, forward_parents, std::back_inserter(path));
    std::reverse(path.begin(), path.end());
    // remove middle node
    assert(path.size() > 0);
    path.pop_back();
    detail::backtrack_path<GraphT>(middle, reverse_parents, std::back_inserter(path));

    (void)start;
    (void)target;
    assert(path.front() == start);
    assert(path.back() == target);

    return path;
}

template <typename GraphT>
auto get_path(const typename GraphT::node_id_t start,
              const typename GraphT::node_id_t target,
              const ParentVector<GraphT> &parents)
{
    std::vector<typename CostVector<GraphT>::value_t> path_labels;
    std::vector<typename GraphT::node_id_t> path;
    detail::backtrack_path<GraphT>(target, parents, std::back_inserter(path));
    std::reverse(path.begin(), path.end());

    (void)start;
    (void)target;
    assert(path.front() == start);
    assert(path.back() == target);

    return path;
}
} // namespace gpusssp::common

#endif
