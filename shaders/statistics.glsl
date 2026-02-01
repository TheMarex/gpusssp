// GLSL header for GPU statistics collection
// This file can be included in compute shaders to enable statistics counting

#ifndef GPUSSSP_GPU_STATISTICS_GLSL
#define GPUSSSP_GPU_STATISTICS_GLSL

#define STAT_NEARFAR_RELAX_EDGES 0
#define STAT_NEARFAR_RELAX_IMPROVED 1
#define STAT_NEARFAR_COMPACT_NODES 2
#define STAT_DELTASTEP_EDGES_RELAXED 3
#define STAT_DELTASTEP_EDGES_IMPROVED 4
#define STAT_BELLMANFORD_EDGES_RELAXED 5
#define STAT_BELLMANFORD_EDGES_IMPROVED 6
#define STAT_NUM_EVENTS 7

#ifdef ENABLE_STATISTICS
    #define statisticsCount(counter_id) atomicAdd(statistics_counters[counter_id], 1)
    #define statisticsAdd(counter_id, value) atomicAdd(statistics_counters[counter_id], value)
#else
    #define statisticsCount(counter_id)
    #define statisticsAdd(counter_id, value)
#endif

#endif
