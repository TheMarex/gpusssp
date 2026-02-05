#ifndef GPUSSSP_COMMON_ZORDER_HPP
#define GPUSSSP_COMMON_ZORDER_HPP

#include "common/coordinate.hpp"

#include <cstdint>
#include <immintrin.h>

namespace gpusssp
{
namespace common
{

inline uint64_t morton_encode(uint32_t x, uint32_t y)
{
    return _pdep_u64(x, 0x5555555555555555ULL) | _pdep_u64(y, 0xAAAAAAAAAAAAAAAAULL);
}

// Converts a coordinate to a z-order curve value
// Offsets coordinates to make them positive, then interleaves bits
inline std::uint64_t coordinate_to_zorder(const Coordinate &coord)
{
    constexpr std::int64_t lon_offset = static_cast<std::int64_t>(180 * Coordinate::PRECISION);
    constexpr std::int64_t lat_offset = static_cast<std::int64_t>(90 * Coordinate::PRECISION);

    std::uint32_t lon_positive = static_cast<std::uint32_t>(coord.lon + lon_offset);
    std::uint32_t lat_positive = static_cast<std::uint32_t>(coord.lat + lat_offset);

    return morton_encode(lon_positive, lat_positive);
}

} // namespace common
} // namespace gpusssp

#endif
