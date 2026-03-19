# Experiments

This directory contains tools for running and managing experiments on the GPUSSSP algorithms.

## Quick Start

### 1. Create an experiment branch

```bash
./experiments/xps.py create my_experiment
```

This creates a new branch `experiment/my_experiment`.

### 2. Make code changes (optional)

If you need to modify the code for your experiment variant, make the changes and commit them normally:

```bash
# Make your changes
git add include/gpu/deltastep.hpp
git commit -m "Adjust bucket size calculation"
```

### 3. Add instrumentation commit

Add an empty commit that defines what to run:

```bash
./experiments/xps.py add "deltastep,nearfar" "delta=900,1800,3600 data=berlin"
```

This creates an empty commit with the message:
```
xp:my_experiment run:deltastep run:nearfar param:delta=900,1800,3600 param:data=berlin
```

### 4. Run the experiments

```bash
./experiments/xps.py run
```

This will:
- Find all instrumentation commits in the current branch
- Check out each commit
- Build the experiment binaries in Release mode
- Run the experiments
- Return to your branch and commit the results

## Workflow

### Multiple Variants

You can test multiple variants in one experiment:

```bash
# Create branch
./experiments/xps.py create bucket_comparison

# Variant 1: baseline
./experiments/xps.py add "deltastep" "delta=900 data=berlin"

# Variant 2: modify code
# ... edit code ...
git add .
git commit -m "Double bucket count"
./experiments/xps.py add "deltastep" "delta=900 data=berlin"

# Variant 3: modify code again
# ... edit code ...
git add .
git commit -m "Halve bucket count"
./experiments/xps.py add "deltastep" "delta=900 data=berlin"

# Run all variants
./experiments/xps.py run
```

Each instrumentation commit will be checked out and run independently.

## Command Reference

### `xps.py create <xp_name>`

Creates a new experiment branch named `experiment/<xp_name>`.

**Example:**
```bash
./experiments/xps.py create delta_sweep
```

### `xps.py add <run_targets> <params> [metrics]`

Adds an empty instrumentation commit to the current experiment branch.

**Run Targets:** Comma-separated list of:
- `dijkstra` - CPU Dijkstra baseline
- `deltastep` - GPU Delta-Stepping
- `bellmanford` - GPU Bellman-Ford
- `nearfar` - GPU Near-Far
- `dial` - CPU Dial's algorithm (uses BucketQueue)

**Parameters:** Space-separated `key=value` pairs:
- `data=<name>` - Dataset name (required, e.g., `berlin`)
- `delta=<values>` - Delta values, comma-separated for multiple (for deltastep, nearfar)
- `batch_size=<value>` - Batch size for GPU processing (for deltastep, nearfar)
- `range=<value>` - Bucket range (for dial, default: 32768)
- `gpu=<id>` - GPU device ID (optional)

**Metrics:** Optional third argument. Default is `time`. Use `edges_relaxed` for edge-count metrics. Note: `time` and requires a different build configurations and cannot be mixed with other metrics.

**Examples:**
```bash
./experiments/xps.py add "deltastep" "delta=900 data=berlin"
./experiments/xps.py add "deltastep,nearfar" "delta=900,1800,3600 data=berlin"
./experiments/xps.py add "deltastep" "delta=900 data=berlin batch_size=128"
./experiments/xps.py add "deltastep" "delta=900 data=berlin" "edges_relaxed"
```

### `xps.py run`

Runs all experiments in the current branch. Must be run from an `experiment/*` branch.

**Example:**
```bash
./experiments/xps.py run
```

### `xps.py compare [xp_name] [...]`

Compare experiment results and print summary tables. If `xp_name` is omitted, shows results for all experiments in `experiments/results/`. Run `./experiments/xps.py compare --help` for options.

**Example:**
```bash
./experiments/xps.py compare my_experiment --variant deltastep --device 0123abcd --metrics time,edges_relaxed
```

### `xps.py plot [xp_name] [...]`

Generate visualization plots (histograms and Dijkstra rank boxplots) for experiment results. Plots are saved in the respective results directories. Run `./experiments/xps.py plot --help` for options.

**Example:**
```bash
./experiments/xps.py plot my_experiment --variant nearfar --device 0123abcd
```

### `xps.py grid [xp_name] [...]`

Show grids comparing parameter combinations: best secondary-param per primary-param/rank, and speedup for combos. Run `./experiments/xps.py grid --help` for options.

**Example:**
```bash
./experiments/xps.py grid my_experiment --primary-param delta --secondary-param batch_size --show top3 --variant deltastep
```

### `xps.py update [xp_name]`

Rebase the experiment branch on `main` and drop any previous result commits. This is useful for cleaning up a branch before a fresh run or after fixing code issues.

**Example:**
```bash
./experiments/xps.py update
```

## Files

- `xps.py` - Main experiment management tool

- `experiment_*.cpp` - Experiment runner executables
- `results/` - Generated experiment results (git-ignored except after committed by `xps.py run`)
- `evaluation/` - Analysis and validation tools
- `plots/` - Plotting utilities

## Tips

- Experiment names should use lowercase and underscores only
- Each instrumentation commit is run independently (code changes before it are included)
- Results are automatically committed after a successful run
- Always use `data=berlin_zorder` or whatever dataset you have preprocessed in `cache/`
- For meaningful performance comparison, experiments are always built in Release mode
