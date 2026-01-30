#ifndef GPUSSSP_COMMON_CSV_PARSER_HPP
#define GPUSSSP_COMMON_CSV_PARSER_HPP

#include "common/string_util.hpp"

#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace gpusssp::common
{

namespace csv
{
struct skip final
{
    bool operator==(const skip &) const { return true; }
};
inline skip ignored;
} // namespace csv

namespace detail
{
// Helper to detect if a type is std::optional
template <typename T> struct is_optional : std::false_type
{
};

template <typename T> struct is_optional<std::optional<T>> : std::true_type
{
};

template <typename T> inline constexpr bool is_optional_v = is_optional<T>::value;

// Helper to extract the inner type from std::optional
template <typename T> struct optional_inner_type
{
    using type = T;
};

template <typename T> struct optional_inner_type<std::optional<T>>
{
    using type = T;
};

template <typename T> using optional_inner_type_t = typename optional_inner_type<T>::type;

// Note: We don't do exhaustive compile-time validation of optional positions
// as it causes excessive template instantiations. The runtime behavior will
// work correctly: if a non-optional column follows an optional one and CSV
// data is missing columns, it will throw a runtime error when trying to parse
// the missing required column.

template <std::size_t Column, class... Types>
void parse_column(std::vector<std::string> &columns, std::tuple<Types...> &tuple)
{
    using column_type = typename std::tuple_element<Column, std::tuple<Types...>>::type;
    using inner_type = optional_inner_type_t<column_type>;

    // Check if we have enough columns
    if (Column >= columns.size())
    {
        // If this column is optional, set it to nullopt and continue
        if constexpr (is_optional_v<column_type>)
        {
            std::get<Column>(tuple) = std::nullopt;
        }
        else
        {
            throw std::runtime_error("Missing required column at index " + std::to_string(Column));
        }
    }
    else
    {
        // We have the column, parse it
        // clang-format off
        if constexpr (is_optional_v<column_type>) {
            // Parse into the optional type
            if constexpr(std::is_same_v<double, inner_type>) {
                std::get<Column>(tuple) = std::stof(columns[Column]);
            } else if constexpr(std::is_same_v<int, inner_type>) {
                std::get<Column>(tuple) = std::stoi(columns[Column]);
            } else if constexpr(std::is_same_v<unsigned, inner_type>) {
                std::get<Column>(tuple) = std::stoi(columns[Column]);
            } else if constexpr(std::is_same_v<unsigned char, inner_type>) {
                std::get<Column>(tuple) = static_cast<unsigned char>(std::stoi(columns[Column]));
            } else if constexpr(std::is_same_v<std::string, inner_type>) {
                std::get<Column>(tuple) = columns[Column];
            } else if constexpr(std::is_same_v<csv::skip, inner_type>) {
                std::get<Column>(tuple) = csv::ignored;
            } else {
                throw std::runtime_error("Can't parse type in optional.");
            }
        } else if constexpr(std::is_same_v<double, column_type>) {
            std::get<Column>(tuple) = std::stof(columns[Column]);
        } else if constexpr(std::is_same_v<int, column_type>) {
            std::get<Column>(tuple) = std::stoi(columns[Column]);
        } else if constexpr(std::is_same_v<unsigned, column_type>) {
            std::get<Column>(tuple) = std::stoi(columns[Column]);
        } else if constexpr(std::is_same_v<unsigned char, inner_type>) {
            std::get<Column>(tuple) = static_cast<unsigned char>(std::stoi(columns[Column]));
        } else if constexpr(std::is_same_v<std::string, column_type>) {
            std::get<Column>(tuple).swap(columns[Column]);
        } else if constexpr(std::is_same_v<csv::skip, column_type>) {
            std::get<Column>(tuple) = csv::ignored;
        } else {
            throw std::runtime_error("Can't parse type.");
        }
        // clang-format on
    }

    // clang-format off
    if constexpr(Column + 1 < std::tuple_size<std::tuple<Types...>>::value) {
        parse_column<Column + 1>(columns, tuple);
    }
    // clang-format on
}

// Count the number of required (non-optional) columns
template <std::size_t Column, class... Types> struct count_required_columns_impl;

// Base case: reached end of types
template <std::size_t Column, class... Types>
struct count_required_columns_impl<Column, std::tuple<Types...>>
{
    static constexpr std::size_t value = 0;
};

// Recursive case
template <class T, class... Rest> struct count_required_columns_impl<0, std::tuple<T, Rest...>>
{
    static constexpr std::size_t current = is_optional_v<T> ? 0 : 1;
    static constexpr std::size_t value =
        current + count_required_columns_impl<0, std::tuple<Rest...>>::value;
};

template <class... Types> struct count_required_columns
{
    static constexpr std::size_t value =
        count_required_columns_impl<0, std::tuple<Types...>>::value;
};

template <class... Types> std::tuple<Types...> parse_columns(std::vector<std::string> &columns)
{
    std::tuple<Types...> tuple;

    // resursively fills the tuple
    parse_column<0>(columns, tuple);

    return tuple;
}

template <typename T> auto to_csv_column(const T &value);

template <typename T> struct stream_printer
{
    friend auto &operator<<(std::ostream &ss, const stream_printer &printer)
    {
        ss << printer.value;
        return ss;
    }

    T value;
};

template <typename T1, typename T2> struct stream_printer<std::tuple<T1, T2>>
{
    friend auto &operator<<(std::ostream &ss, const stream_printer &other)
    {
        ss << "(" << std::get<0>(other.value) << " " << std::get<1>(other.value) << ")";
        return ss;
    }

    std::tuple<T1, T2> value;
};

template <typename T> struct stream_printer<std::vector<T>>
{
    friend auto &operator<<(std::ostream &ss, const stream_printer &other)
    {
        if (other.value.empty())
            return ss;

        for (auto iter = other.value.begin(); iter != std::prev(other.value.end()); ++iter)
            ss << to_csv_column(*iter) << ",";
        ss << to_csv_column(*std::prev(other.value.end()));
        return ss;
    }

    std::vector<T> value;
};

template <> struct stream_printer<unsigned char>
{
    friend auto &operator<<(std::ostream &ss, const stream_printer &other)
    {
        ss << (unsigned)other.value;
        return ss;
    }

    unsigned char value;
};

template <typename T> struct stream_printer<std::optional<T>>
{
    friend auto &operator<<(std::ostream &ss, const stream_printer &other)
    {
        if (!other.value.has_value())
            return ss;

        ss << to_csv_column(*other.value);
        return ss;
    }

    std::optional<T> value;
};

template <> struct stream_printer<csv::skip>
{
    friend auto &operator<<(std::ostream &ss, const stream_printer &) { return ss; }
};

template <typename T> auto to_csv_column(const T &value)
{
    return stream_printer<typename std::remove_reference<T>::type>{value};
}

template <std::size_t Column, class... Types>
void append_column(std::stringstream &ss, const std::tuple<Types...> &tuple, const char *delimiter)
{
    // clang-format off
    if constexpr(Column != 0) {
        ss << delimiter;
    }
    // clang-format on

    ss << to_csv_column(std::get<Column>(tuple));

    // clang-format off
    if constexpr(Column + 1 < std::tuple_size<std::tuple<Types...>>::value) {
        append_column<Column + 1>(ss, tuple, delimiter);
    }
    // clang-format on
}

template <class... Types>
std::string join_tuple(const std::tuple<Types...> &tuple, const char *delimiter)
{
    std::stringstream ss;

    append_column<0>(ss, tuple, delimiter);

    return ss.str();
}
} // namespace detail

template <class... Types> class CSVReader
{
  public:
    using output_t = std::tuple<Types...>;

    CSVReader(const std::string &path, const char *delimiter = ",")
        : stream(path), delimiter(delimiter)
    {
        stream.exceptions(std::ifstream::badbit);
    }

    bool read_header(std::vector<std::string> &header)
    {
        std::string line;
        if (std::getline(stream, line))
        {
            detail::split(header, line, delimiter);
            return true;
        }

        return false;
    }

    bool read(output_t &value)
    {
        std::string line;
        if (std::getline(stream, line))
        {
            std::vector<std::string> tokens;
            detail::split(tokens, line, delimiter);

            // ignore empty lines
            if (tokens.size() == 0)
                return true;

            constexpr std::size_t required_columns =
                detail::count_required_columns<Types...>::value;
            constexpr std::size_t total_columns = std::tuple_size<output_t>::value;

            // Check if we have too few columns (less than required)
            if (tokens.size() < required_columns)
            {
                throw std::runtime_error("Could not parse line: \"" + line +
                                         "\". Number of columns is " +
                                         std::to_string(tokens.size()) + " but at least " +
                                         std::to_string(required_columns) + " are required.");
            }

            // Check if we have too many columns
            if (tokens.size() > total_columns)
            {
                throw std::runtime_error("Could not parse line: \"" + line +
                                         "\". Number of columns is " +
                                         std::to_string(tokens.size()) + " but at most " +
                                         std::to_string(total_columns) + " are expected.");
            }

            value = detail::parse_columns<Types...>(tokens);
            return true;
        }

        return false;
    }

  private:
    const char *delimiter;
    std::ifstream stream;
};

template <class... Types> class CSVWriter
{
  public:
    using input_t = std::tuple<Types...>;

    CSVWriter(const std::string &path, const char *delimiter = ",")
        : stream(path), delimiter(delimiter)
    {
        stream.exceptions(std::ifstream::badbit);
    }

    void write_header(const std::vector<std::string> &header)
    {
        stream << detail::join(header, delimiter) << std::endl;
        ;
    }

    void write(const input_t &value)
    {
        stream << detail::join_tuple<Types...>(value, delimiter) << std::endl;
    }

  private:
    const char *delimiter;
    std::ofstream stream;
};
} // namespace gpusssp::common

#endif
