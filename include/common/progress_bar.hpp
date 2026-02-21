#ifndef GPUSSSP_COMMON_PROGRESS_BAR_HPP
#define GPUSSSP_COMMON_PROGRESS_BAR_HPP

#include "common/logger.hpp"
#include <iomanip>
#include <iostream>

namespace gpusssp::common
{

class ProgressBar
{
    static const constexpr size_t WIDTH = 40;

  public:
    explicit ProgressBar(size_t total) : total_count(total) { update(0); }

    void update(size_t count)
    {
        auto &os = Logger::get().log(LogLevel::INFO);
        if (os.rdbuf() == nullptr)
        {
            return;
        }

        current_count = count;
        auto progress = static_cast<float>(current_count) / static_cast<float>(total_count);
        auto pos = static_cast<size_t>(static_cast<float>(WIDTH) * progress);

        os << "\r[";
        for (size_t i = 0; i < WIDTH; ++i)
        {
            if (i < pos)
            {
                os << "=";
            }
            else if (i == pos && current_count < total_count)
            {
                os << ">";
            }
            else
            {
                os << " ";
            }
        }
        os << "] " << std::setw(3) << static_cast<int>(progress * 100.0f) << "% (" << current_count
           << "/" << total_count << ")";

        if (current_count >= total_count)
        {
            os << '\n';
        }
        else
        {
            os.flush();
        }
    }

    void increment() { update(current_count + 1); }

  private:
    size_t total_count;
    size_t current_count = 0;
};

} // namespace gpusssp::common

#endif
