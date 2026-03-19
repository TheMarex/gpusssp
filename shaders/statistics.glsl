// GLSL header for GPU statistics collection
// This file can be included in compute shaders to enable statistics counting

#ifndef GPUSSSP_GPU_STATISTICS_GLSL
#define GPUSSSP_GPU_STATISTICS_GLSL

#extension GL_EXT_shader_subgroup_extended_types_int64: enable

#define STAT_NEARFAR_EDGES_RELAXED 0
#define STAT_NEARFAR_EDGES_IMPROVED 1
#define STAT_NEARFAR_NODES_COMPACTED 2
#define STAT_DELTASTEP_EDGES_RELAXED 3
#define STAT_DELTASTEP_EDGES_IMPROVED 4
#define STAT_BELLMANFORD_EDGES_RELAXED 5
#define STAT_BELLMANFORD_EDGES_IMPROVED 6
#define STAT_NUM_EVENTS 7

#ifdef ENABLE_STATISTICS

shared uint64_t local_statistics[STAT_NUM_EVENTS];
uint thread_local_stats[STAT_NUM_EVENTS];

void statisticsInit()
{
    for (uint i = 0; i < STAT_NUM_EVENTS; i++)
    {
        thread_local_stats[i] = 0;
    }
    
    if (gl_LocalInvocationID.x < STAT_NUM_EVENTS)
    {
        local_statistics[gl_LocalInvocationID.x] = 0;
    }
    barrier();
}

#define statisticsCount(counter_id) thread_local_stats[counter_id]++
#define statisticsAdd(counter_id, value) thread_local_stats[counter_id] += (value)

#define statisticsSync(statistics_buffer) \
    for (uint i = 0; i < STAT_NUM_EVENTS; i++) \
    { \
        uint64_t subgroup_sum = subgroupAdd(uint64_t(thread_local_stats[i])); \
        if (subgroupElect()) \
        { \
            atomicAdd(local_statistics[i], subgroup_sum); \
        } \
    } \
    \
    barrier(); \
    \
    if (gl_LocalInvocationID.x < STAT_NUM_EVENTS) \
    { \
        if (local_statistics[gl_LocalInvocationID.x] > 0) \
        { \
            atomicAdd(statistics_buffer[gl_LocalInvocationID.x], \
                     local_statistics[gl_LocalInvocationID.x]); \
        } \
    }

#else

void statisticsInit() {}
#define statisticsCount(counter_id)
#define statisticsAdd(counter_id, value)
#define statisticsSync(statistics_buffer)

#endif

#endif
