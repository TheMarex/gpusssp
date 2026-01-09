#ifndef GPUSSSP_COMMON_STRING_UTIL_HPP
#define GPUSSSP_COMMON_STRING_UTIL_HPP

#include <sstream>
#include <string>
#include <vector>

namespace gpusssp::common::detail
{

inline void
split(std::vector<std::string> &tokens, const std::string &input, const std::string &delimiters)
{
    tokens.clear();
    std::size_t start = 0;
    while (true)
    {
        std::size_t end = input.find_first_of(delimiters, start);
        if (end == std::string::npos)
        {
            tokens.push_back(input.substr(start));
            break;
        }
        tokens.push_back(input.substr(start, end - start));
        start = end + 1;
    }
}

inline std::string join(const std::vector<std::string> &elements, const std::string &delimiter)
{
    if (elements.empty())
    {
        return "";
    }
    std::ostringstream os;
    auto it = elements.begin();
    os << *it++;
    while (it != elements.end())
    {
        os << delimiter << *it++;
    }
    return os.str();
}

} // namespace gpusssp::common::detail

#endif // GPUSSSP_COMMON_STRING_UTIL_HPP
