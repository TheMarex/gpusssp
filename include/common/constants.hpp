#ifndef GPUSSSP_CONSTANTS_H
#define GPUSSSP_CONSTANTS_H

#include <cmath>
#include <cstdint>
#include <limits>

namespace gpusssp
{
namespace common
{

constexpr std::uint32_t INVALID_ID = std::numeric_limits<std::uint32_t>::max();
// can be doubled without overflowing
constexpr std::int32_t INF_WEIGHT = std::numeric_limits<std::int32_t>::max() / 2;

constexpr double FIXED_POINT_RESOLUTION = 10.0;

constexpr double from_fixed(std::int32_t value) { return value / common::FIXED_POINT_RESOLUTION; }

constexpr std::int32_t to_fixed(double value) { return value * common::FIXED_POINT_RESOLUTION; }

inline std::int32_t to_upper_fixed(double value)
{
    return std::ceil(value * common::FIXED_POINT_RESOLUTION);
}

} // namespace common
} // namespace gpusssp

#endif
