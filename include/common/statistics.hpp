#ifndef GPUSSSP_COMMON_STATISTICS_HPP
#define GPUSSSP_COMMON_STATISTICS_HPP

#ifdef ENABLE_STATISTICS
#include "common/irange.hpp"

#include <sstream>
#endif

#include <array>
#include <chrono>
#include <cstdint>
#include <string>

namespace gpusssp
{
namespace common
{

enum class StatisticsEvent : uint8_t
{
    QUEUE_POP,
    QUEUE_PUSH,
    QUEUE_DECREASE_KEY,
    QUEUE_INCREASE_KEY,
    DIJKSTRA_STALL,
    DIJKSTRA_RELAX,
    DELTASTEP_BUCKET,
    DELTASTEP_HEAVY,
    NEARFAR_PHASE,
    NEARFAR_RELAX,
    NEARFAR_INIT_DURATION,
    NEARFAR_RELAX_DURATION,
    NEARFAR_COMPACT_DURATION,
    NUM_EVENTS
};

inline const char *event_to_name(StatisticsEvent name)
{
    static std::array<const char *, static_cast<std::size_t>(StatisticsEvent::NUM_EVENTS)> names{
        {"QUEUE_POP",
         "QUEUE_PUSH",
         "QUEUE_DECREASE_KEY",
         "QUEUE_INCREASE_KEY",
         "DIJKSTRA_STALL",
         "DIJKSTRA_RELAX",
         "DELTASTEP_BUCKET",
         "DELTASTEP_HEAVY",
         "NEARFAR_PHASE",
         "NEARFAR_RELAX",
         "NEARFAR_INIT_DURATION",
         "NEARFAR_RELAX_DURATION",
         "NEARFAR_COMPACT_DURATION"}};
    return names[static_cast<std::size_t>(name)];
}

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

    auto start(StatisticsEvent) { return std::chrono::high_resolution_clock::now(); }
    void stop(StatisticsEvent event, auto start_time)
    {
#ifdef ENABLE_STATISTICS
        auto stop_time = std::chrono::high_resolution_clock::now();
        counts[static_cast<std::size_t>(event)] +=
            std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count();
#else
        (void)event;
        (void)start_time;
#endif
    }

    std::string summary() const
    {
#ifdef ENABLE_STATISTICS
        auto event_counters = counts;
        std::stringstream ss;
        for (const auto event_id : irange<std::uint8_t>(0, event_counters.size()))
        {
            StatisticsEvent event{event_id};
            ss << event_to_name(event) << ": " << event_counters[event_id] << std::endl;
        }
        return ss.str();
#else
        return "";
#endif
    }

  private:
    Statistics() {}

#ifdef ENABLE_STATISTICS
    std::array<uint64_t, static_cast<std::size_t>(StatisticsEvent::NUM_EVENTS)> counts;
#endif
};

} // namespace common
} // namespace gpusssp

#endif
