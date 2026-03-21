#ifndef GPUSSSP_COMMON_CLI_HPP
#define GPUSSSP_COMMON_CLI_HPP

#include "common/coordinate.hpp"
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>

namespace gpusssp::common
{

inline std::optional<Coordinate> parse_coordinate(const std::string &s)
{
    if (s == "random")
    {
        return {};
    }

    auto pos = s.find(',');
    if (pos == std::string::npos)
    {
        return {};
    }
    return Coordinate::from_floating(std::stod(s.substr(0, pos)), std::stod(s.substr(pos + 1)));
}

inline std::optional<uint32_t> parse_node_id(const std::string &s)
{
    if (s.empty() || !std::isdigit(s[0]))
    {
        return {};
    }

    std::size_t pos;
    auto node_id = std::stoi(s, &pos);
    if (pos != s.size())
    {
        return {};
    }

    return node_id;
}

} // namespace gpusssp::common

#endif // GPUSSSP_COMMON_CLI_HPP
