#ifndef GPUSSSP_COMMON_WEB_MERCATOR_HPP
#define GPUSSSP_COMMON_WEB_MERCATOR_HPP

#include <cmath>
#include <numbers>

#include "common/coordinate.hpp"

namespace gpusssp::common
{

struct WebMercatorPoint
{
    double x;
    double y;
};

inline WebMercatorPoint to_web_mercator(const Coordinate &coord)
{
    auto [lon, lat] = coord.to_floating();
    constexpr double EARTH_RADIUS = 6378137.0;

    double x = EARTH_RADIUS * lon * std::numbers::pi / 180.0;

    double lat_rad = lat * std::numbers::pi / 180.0;
    double y = EARTH_RADIUS * std::log(std::tan(std::numbers::pi / 4.0 + lat_rad / 2.0));

    return {x, y};
}

} // namespace gpusssp::common

#endif
