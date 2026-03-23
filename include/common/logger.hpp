#ifndef GPUSSSP_COMMON_LOGGER_HPP
#define GPUSSSP_COMMON_LOGGER_HPP

#include <cstdint>
#include <iostream>
#include <ostream>
#include <streambuf>

namespace gpusssp::common
{

enum class LogLevel : std::uint8_t
{
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3
};

class NullStreambuf : public std::streambuf
{
  protected:
    int overflow(int c) override { return c; }
};

class NullStream : public std::ostream
{
  public:
    NullStream() : std::ostream(&null_buf) {}
    template <typename T> NullStream &operator<<(const T &) { return *this; }
    NullStream &operator<<(std::ostream &(*)(std::ostream &)) { return *this; }

  private:
    NullStreambuf null_buf;
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

    Logger(const Logger &) = delete;
    Logger &operator=(const Logger &) = delete;

  private:
    Logger() = default;

#ifdef NDEBUG
    LogLevel min_level{LogLevel::INFO};
#else
    LogLevel min_level{LogLevel::DEBUG};
#endif
    NullStream null_stream;
};

inline std::ostream &log() { return Logger::get().log(LogLevel::INFO); }

inline std::ostream &log_info() { return Logger::get().log(LogLevel::INFO); }

inline std::ostream &log_warning() { return Logger::get().log(LogLevel::WARNING); }

inline std::ostream &log_error() { return Logger::get().log(LogLevel::ERROR); }

#ifdef NDEBUG
inline NullStream &log_debug()
{
    static NullStream null_stream;
    return null_stream;
}
#else
inline std::ostream &log_debug() { return Logger::get().log(LogLevel::DEBUG); }
#endif

} // namespace gpusssp::common

#endif
