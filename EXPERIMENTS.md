## edge_balancing

Investigate balancing the number of edges processed per thread.

## fixed_dispatch

Compare performance of fixed dispatch (e.g. dispatch to all nodes in the graph) to min/max based on changed ID.

## workgroup_size

Investigate impact of workgroup size on performance.

## delta_sweep

Compare performance DeltaStep / NF with different delta values.

## changed_nodes_compaction

Hypothesis: compacted changed-node dispatch in DeltaStep (commit d0a22bd0) yields lower per-bucket and average runtimes than the reverted min/max-range version on berlin_zorder. Instrument variants on `main` and `main`+`revert d0a22bd0`, running DeltaStep with delta=900,1800,3600 on gpu 0. Success requires the compacted variant to be faster across every bucket and overall; any mixed result counts as a failure.

Outcome: hypothesis invalidated – reverting d0a22bd0 (min/max dispatch) was 25–40% faster across berlin_zorder percentiles and per-bucket averages for delta 900/1800/3600.

## deltastep_no_heavy

Hypothesis: Removing the DeltaStep heavy-pass (max-weight filtered sweep) and scanning all edges in the light-pass shader will retain correctness while lowering per-bucket runtime. Instrument `deltastep` on dataset `berlin_zorder` with `delta=900` via `xps.py` on a dedicated branch. Success requires equal best-distance results and a ≥5% reduction in average bucket runtime compared to the baseline commit; any regression or correctness issue fails the hypothesis.

Outcome: hypothesis validated – removing the heavy pass cut average query time from 22.6 ms to 18.7 ms (delta 900 on berlin_zorder) with matching best distances.
