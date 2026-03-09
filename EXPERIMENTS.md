## bucket_queue_no_free_list

Hypothesis: removing `free_entries` from `BucketQueue` (replacing the free-list recycling of `BucketEntry` slots with a simple grow-only `entries` vector that is reset on `clear()`) reduces overhead of Dial's algorithm.
Outcome: hypothesis invalidated – berlin_zorder Dial queries slowed down by 4-11% without the `free_entries` recycling.

## bucket_queue_plain_id_entry

Hypothesis: replacing the `BucketQueue::id_entry` `LazyClearVector<unsigned>` with a plain `std::vector<unsigned>` eliminates the generation check and indirection overhead on every access. Even though `clear()` must loop over `entries` to reset `id_entry[entry.p.id] = INVALID_ID`, we expect Dial on berlin_zorder to run faster overall.
Outcome: hypothesis invalidated – on bayern_zorder the plain vector variant was 1.1–1.7× slower for ranks 0–16 and only caught up at the highest ranks.

## edge_balancing

Hypothesis: Balancing the number of edges processed per thread improves performance.
Outcome: Edge balancing does not yield a significant speed-up for road networks.

## fixed_dispatch

Hypothesis: Dispatching deltastep to min/max based on changed ID vs dispatching to all nodes in the graph improves performance.
Outcome: min/max changed ID is faster.

## workgroup_size

Hypothesis: The shader workgroup size impacts the performance of the nearfar/deltastep algorithms.
Outcome: The workgroup size is largely irrelevant.

## delta_sweep

Hypothesis: Performance of Deltastep / Nearfar highly depends on delta values.

## changed_nodes_compaction

Hypothesis: compacted changed-node dispatch in DeltaStep (commit d0a22bd0) runtimes berlin_zorder.
Outcome: hypothesis invalidated – reverting d0a22bd0 (min/max dispatch) was 25–40% faster across berlin_zorder percentiles and per-bucket averages for delta 900/1800/3600.

## deltastep_no_heavy

Hypothesis: Removing the DeltaStep heavy-pass (max-weight filtered sweep) and scanning all edges in the light-pass shader will retain correctness while lowering per-bucket runtime.
Outcome: hypothesis validated – removing the heavy pass cut average query time from 22.6 ms to 18.7 ms (delta 900 on berlin_zorder).

## heatmap

Hypothesis: Z-order sorting improves the memory locality of buffer access.
Outcome: This is true memory access patterns are way more compact, for "natural" node ordering they are essentially random.

## rangesize

Hypothesis: Z-order descreases the min_changed_id .. max_changed_id size.
Outcome: This is true especially in the early phases of the search and in the late stages for each bucket.

## boxplot_delta900_berlin_germany

Hypothesis: Dijkstra should clearly lead on berlin with NearFar second, while on germany NearFar and Dijkstra stay close and DeltaStep is the slowest.
Outcome: On berlin_zorder p50 Dijkstra finished in 19 µs versus 11.7 ms for DeltaStep and 19.5 ms for NearFar (DeltaStep beat NearFar), and on germany_zorder Dijkstra’s p50 was 234 µs compared to 65 ms for NearFar and 133 ms for DeltaStep.

## node_batching

Hypothesis: batching multiple nodes per DeltaStep invocation reduces per-bucket runtime on berlin_zorder by amortizing control overhead. We will compare SPECIALIZATION_NODES_PER_INVOCATION values {1,4,8,16,32} with delta 900 on GPU 0.
Outcome: Invalidated. berlin_zorder p50 runtimes at delta 900 worsened for 4–16 nodes/invocation (up to +10%), while 32 nodes/invocation only improved p50 from 10.8 ms to 10.7 ms and left high percentiles unchanged.

## prerecorded_cmd_buf

Hypothesis: Pre-recording command buffers significantly reduces CPU overhead during the main loop and yield better runtimes for deltastep and nearfar.
Outcome: Speedup for deltastep between 1.52 and 1.74 and for nearfar between 1.27 and 1.32 on berlin -> way faster.
