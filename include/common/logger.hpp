#ifndef GPUSSSP_COMMON_LOGGER_HPP
#define GPUSSSP_COMMON_LOGGER_HPP

#include <iostream>
#include <ostream>

namespace gpusssp
{
namespace common
{

enum class LogLevel
{
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3
};

class NullStream : public std::ostream
{
  public:
    NullStream() : std::ostream(nullptr) {}
    template <typename T> NullStream &operator<<(const T &) { return *this; }
    NullStream &operator<<(std::ostream &(*)(std::ostream &)) { return *this; }
};

class Logger
{
  public:
    static Logger &get();

    static void set_level(LogLevel level) { get().min_level = level; }

    static LogLevel get_level() { return get().min_level; }

    std::ostream &log(LogLevel level)
    {
        if (level >= min_level)
        {
            return std::cerr;
        }
        return null_stream;
    }

  private:
    Logger()
#ifdef NDEBUG
        : min_level(LogLevel::INFO)
#else
        : min_level(LogLevel::DEBUG)
#endif
    {
    }

    Logger(const Logger &) = delete;
    Logger &operator=(const Logger &) = delete;

    LogLevel min_level;
    NullStream null_stream;
};

inline std::ostream &log() { return Logger::get().log(LogLevel::INFO); }

inline std::ostream &log_info() { return Logger::get().log(LogLevel::INFO); }

inline std::ostream &log_warning() { return Logger::get().log(LogLevel::WARNING); }

inline std::ostream &log_error() { return Logger::get().log(LogLevel::ERROR); }

#ifdef NDEBUG
inline NullStream log_debug() { return NullStream(); }
#else
inline std::ostream &log_debug() { return Logger::instance().log(LogLevel::DEBUG); }
#endif

} // namespace common
} // namespace gpusssp

#endif
