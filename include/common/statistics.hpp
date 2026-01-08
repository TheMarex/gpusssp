#ifndef GPUSSSP_COMMON_STATISTICS_HPP
#define GPUSSSP_COMMON_STATISTICS_HPP

#include <vector>

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
        // Implementation can be empty or simple counter
        // counts[static_cast<int>(event)]++;
    }

  private:
    Statistics() {}
};

} // namespace common
} // namespace gpusssp

#endif
