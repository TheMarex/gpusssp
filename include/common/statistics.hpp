#ifndef GPUSSSP_COMMON_STATISTICS_HPP
#define GPUSSSP_COMMON_STATISTICS_HPP

#ifdef ENABLE_STATISTICS
#include <array>
#endif

namespace gpusssp
{
namespace common
{

enum class StatisticsEvent
{
    QUEUE_POP,
    QUEUE_PUSH,
    QUEUE_DECREASE_KEY,
    QUEUE_INCREASE_KEY,
    DIJKSTRA_STALL,
    DIJKSTRA_RELAX,
    NUM_EVENTS
};

class Statistics
{
  public:
    static Statistics &get()
    {
        static Statistics instance;
        return instance;
    }

    void count(StatisticsEvent event)
    {
#ifdef ENABLE_STATISTICS
        counts[static_cast<std::size_t>(event)]++;
#else
        (void)event;
#endif
    }

  private:
    Statistics() {}

#ifdef ENABLE_STATISTICS
    std::array<long, static_cast<std::size_t>(StatisticsEvent::NUM_EVENTS)> counts;
#endif
};

} // namespace common
} // namespace gpusssp

#endif
