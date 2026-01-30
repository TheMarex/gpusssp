#ifndef GPUSSSP_COMMON_BENCHMARK_HPP
#define GPUSSSP_COMMON_BENCHMARK_HPP

namespace gpusssp::common
{

template <typename T> inline void DoNotOptimize(T const &value)
{
#if defined(__clang__) || defined(__GNUC__)
    asm volatile("" : : "r,m"(value) : "memory");
#else
#error "Only GCC and clang are supported"
#endif
}

} // namespace gpusssp::common

#endif
