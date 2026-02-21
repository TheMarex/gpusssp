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
