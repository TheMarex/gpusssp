#ifndef GPUSSSP_COMMON_ZORDER_HPP
#define GPUSSSP_COMMON_ZORDER_HPP

#include "common/coordinate.hpp"

#include <cstdint>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace gpusssp::common
{

static inline uint64_t spread_bits(uint32_t x)
{
    uint64_t r = x;
    r = (r | (r << 16)) & 0x0000FFFF0000FFFFULL;
    r = (r | (r << 8)) & 0x00FF00FF00FF00FFULL;
    r = (r | (r << 4)) & 0x0F0F0F0F0F0F0F0FULL;
    r = (r | (r << 2)) & 0x3333333333333333ULL;
    r = (r | (r << 1)) & 0x5555555555555555ULL;
    return r;
}

inline uint64_t morton_encode(uint32_t x, uint32_t y)
{
#if defined(__BMI2__) && (defined(__x86_64__) || defined(__i386__))
    return _pdep_u64(x, 0x5555555555555555ULL) | _pdep_u64(y, 0xAAAAAAAAAAAAAAAAULL);
#elif defined(__aarch64__) || defined(__arm64__)
    return spread_bits(x) | (spread_bits(y) << 1);
#else
#error "morton_encode requires BMI2 (x86) or ARM64"
#endif
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
