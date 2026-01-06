#ifndef GPUSSSP_COMMON_INTEGER_RANGE_HPP
#define GPUSSSP_COMMON_INTEGER_RANGE_HPP

#include <ranges>

namespace gpusssp
{
namespace common
{

template <std::integral I> auto irange(I begin, I end) { return std::views::iota(begin, end); }

} // namespace common
} // namespace gpusssp

#endif // INTEGER_RANGE_HPP
