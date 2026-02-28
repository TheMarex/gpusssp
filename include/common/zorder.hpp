#ifndef GPUSSSP_COMMON_ZORDER_HPP
#define GPUSSSP_COMMON_ZORDER_HPP

#include "common/coordinate.hpp"

#include <cstdint>
#include <immintrin.h>

namespace gpusssp::common
{

inline uint64_t morton_encode(uint32_t x, uint32_t y)
{
    return _pdep_u64(x, 0x5555555555555555ULL) | _pdep_u64(y, 0xAAAAAAAAAAAAAAAAULL);
}

// Converts a coordinate to a z-order curve value
// Offsets coordinates to make them positive, then interleaves bits
inline std::uint64_t coordinate_to_zorder(const Coordinate &coord)
{
    constexpr auto LON_OFFSET = static_cast<std::int64_t>(180 * Coordinate::PRECISION);
    constexpr auto LAT_OFFSET = static_cast<std::int64_t>(90 * Coordinate::PRECISION);

    auto lon_positive = static_cast<std::uint32_t>(coord.lon + LON_OFFSET);
    auto lat_positive = static_cast<std::uint32_t>(coord.lat + LAT_OFFSET);

    return morton_encode(lon_positive, lat_positive);
}

} // namespace gpusssp::common

#endif
