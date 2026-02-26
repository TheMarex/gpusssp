## edge_balancing

Investigate balancing the number of edges processed per thread.

## fixed_dispatch

Compare performance of fixed dispatch (e.g. dispatch to all nodes in the graph) to min/max based on changed ID.

## workgroup_size

Investigate impact of workgroup size on performance.

## delta_sweep

Compare performance DeltaStep / NF with different delta values.

## changed_nodes_compaction

Hypothesis: compacted changed-node dispatch in DeltaStep (commit d0a22bd0) runtimes berlin_zorder.

Outcome: hypothesis invalidated – reverting d0a22bd0 (min/max dispatch) was 25–40% faster across berlin_zorder percentiles and per-bucket averages for delta 900/1800/3600.

## deltastep_no_heavy

Hypothesis: Removing the DeltaStep heavy-pass (max-weight filtered sweep) and scanning all edges in the light-pass shader will retain correctness while lowering per-bucket runtime.

Outcome: hypothesis validated – removing the heavy pass cut average query time from 22.6 ms to 18.7 ms (delta 900 on berlin_zorder).

## heatmap

Hypothesis: Z-order sorting improves the memory locality of buffer access.
This is true memory access patterns are way more compact, for "natural" node ordering they are essentially random.

## rangesize

Hypothesis: Z-order descreases the min_changed_id .. max_changed_id size.
This is true especially in the early phases of the search and in the late stages for each bucket.

## boxplot_delta900_berlin_germany

Compare Dijkstra, DeltaStep, and NearFar at delta 900 on the berlin and germany caches using GPU 0 to capture boxplot comparisons. Hypothesis: Dijkstra should clearly lead on berlin with NearFar second, while on germany NearFar and Dijkstra stay close and DeltaStep is the slowest.
Outcome: hypothesis invalidated – on berlin_zorder p50 Dijkstra finished in 19 µs versus 11.7 ms for DeltaStep and 19.5 ms for NearFar (DeltaStep beat NearFar), and on germany_zorder Dijkstra’s p50 was 234 µs compared to 65 ms for NearFar and 133 ms for DeltaStep.
