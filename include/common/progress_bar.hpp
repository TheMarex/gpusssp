#ifndef GPUSSSP_COMMON_PROGRESS_BAR_HPP
#define GPUSSSP_COMMON_PROGRESS_BAR_HPP

#include "common/logger.hpp"

#include <atomic>
#include <iomanip>
#include <iostream>
#include <mutex>

namespace gpusssp::common
{

class ProgressBar
{
    static const constexpr size_t WIDTH = 40;
    static const constexpr float MIN_INCREMENT = 0.01;

  public:
    explicit ProgressBar(size_t total) : total_count(total) { update(0); }

    void update(size_t count)
    {
        std::scoped_lock lock(m_mutex);
        current_count.store(count);

        auto &os = Logger::get().log(LogLevel::INFO);
        if (os.rdbuf() == nullptr)
        {
            return;
        }

        auto prev_progress = static_cast<float>(prev_update) / static_cast<float>(total_count);
        auto progress = static_cast<float>(count) / static_cast<float>(total_count);
        if (progress - prev_progress < MIN_INCREMENT && count < total_count)
        {
            return;
        }

        auto pos = static_cast<size_t>(static_cast<float>(WIDTH) * progress);

        os << "\r[";
        for (size_t i = 0; i < WIDTH; ++i)
        {
            if (i < pos)
            {
                os << "=";
            }
            else if (i == pos && count < total_count)
            {
                os << ">";
            }
            else
            {
                os << " ";
            }
        }
        os << "] " << std::setw(3) << static_cast<int>(progress * 100.0f) << "% (" << count << "/"
           << total_count << ")";

        if (count >= total_count)
        {
            os << '\n';
        }
        else
        {
            os.flush();
        }

        prev_update = count;
    }

    void increment() { update(current_count.fetch_add(1) + 1); }

  private:
    size_t total_count;
    std::atomic<size_t> current_count{0};
    size_t prev_update = 0;
    std::mutex m_mutex;
};

} // namespace gpusssp::common

#endif
