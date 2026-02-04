#ifndef GPUSSSP_COMMON_ZORDER_HPP
#define GPUSSSP_COMMON_ZORDER_HPP

#include "common/coordinate.hpp"

#include <cstdint>

namespace gpusssp
{
namespace common
{

// Interleaves the bits of two 32-bit unsigned integers into a single 64-bit value
// Pattern: x[0]y[0]x[1]y[1]...x[31]y[31]
inline std::uint64_t interleave_bits(std::uint32_t x, std::uint32_t y)
{
    std::uint64_t result = 0;

    for (int i = 0; i < 32; ++i)
    {
        // Extract bit i from x and y
        std::uint64_t x_bit = (x >> i) & 1ULL;
        std::uint64_t y_bit = (y >> i) & 1ULL;

        // Place x_bit at position 2*i and y_bit at position 2*i+1
        result |= (x_bit << (2 * i)) | (y_bit << (2 * i + 1));
    }

    return result;
}

// Converts a coordinate to a z-order curve value
// Offsets coordinates to make them positive, then interleaves bits
inline std::uint64_t coordinate_to_zorder(const Coordinate &coord)
{
    constexpr std::int64_t lon_offset = static_cast<std::int64_t>(180 * Coordinate::PRECISION);
    constexpr std::int64_t lat_offset = static_cast<std::int64_t>(90 * Coordinate::PRECISION);

    std::uint32_t lon_positive = static_cast<std::uint32_t>(coord.lon + lon_offset);
    std::uint32_t lat_positive = static_cast<std::uint32_t>(coord.lat + lat_offset);

    return interleave_bits(lon_positive, lat_positive);
}

} // namespace common
} // namespace gpusssp

#endif
