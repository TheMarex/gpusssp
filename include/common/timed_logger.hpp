#ifndef GPUSSSP_TIMED_LOGGER_HPP
#define GPUSSSP_TIMED_LOGGER_HPP

#include <chrono>
#include <iostream>
#include <string>

namespace gpusssp::common
{
struct TimedLogger
{
    explicit TimedLogger(const std::string &msg) : start(std::chrono::high_resolution_clock::now())
    {
        std::cerr << msg << "... " << std::flush;
    }

    void finished() const
    {
        auto end = std::chrono::high_resolution_clock::now();
        auto diff = end - start;
        std::cerr << std::chrono::duration_cast<std::chrono::microseconds>(diff).count() / 1000.
                  << " ms." << '\n';
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};
} // namespace gpusssp::common

#endif
