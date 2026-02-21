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

### `xps.py add <run_targets> <params>`

Adds an empty instrumentation commit to the current experiment branch.

**Run Targets:** Comma-separated list of:
- `dijkstra` - CPU Dijkstra baseline
- `deltastep` - GPU Delta-Stepping
- `bellmanford` - GPU Bellman-Ford
- `nearfar` - GPU Near-Far (alternative name for a variant)

**Parameters:** Space-separated `key=value` pairs:
- `delta=<values>` - Delta values (comma-separated for multiple)
- `data=<name>` - Dataset name (required, e.g., `berlin`)
- `gpu=<id>` - GPU device ID (optional, default 0)

**Examples:**
```bash
# Single algorithm, single delta
./experiments/xps.py add "deltastep" "delta=900 data=berlin"

# Multiple algorithms
./experiments/xps.py add "deltastep,nearfar" "delta=900 data=berlin"

# Multiple deltas
./experiments/xps.py add "deltastep" "delta=900,1800,3600 data=berlin"

# Specific GPU
./experiments/xps.py add "deltastep" "delta=900 data=berlin gpu=1"
```

### `xps.py run`

Runs all experiments in the current branch. Must be run from an `experiment/*` branch.

**Example:**
```bash
./experiments/xps.py run
```

### `xps.py compare [xp_name] [--device <hash>] [--variant <substring>]`

Compare experiment results and print summary tables. If `xp_name` is omitted, it shows results for all experiments found in `experiments/results/`.

**Options:**
- `--device <hash>`: Filter results by specific device hash.
- `--variant <substring>`: Filter variants by substring match (e.g., `deltastep`).

**Example:**
```bash
# Compare current experiment
./experiments/xps.py compare

# Compare a specific experiment
./experiments/xps.py compare my_experiment

# Compare only deltastep variants on a specific device
./experiments/xps.py compare --variant deltastep --device 0123abcd
```

### `xps.py plot [xp_name] [--device <hash>] [--variant <substring>]`

Generate visualization plots (histograms and Dijkstra rank boxplots) for experiment results. Plots are saved in the respective results directories.

**Options:**
- `--device <hash>`: Filter results by specific device hash.
- `--variant <substring>`: Filter variants by substring match (e.g., `nearfar`).

**Example:**
```bash
# Generate plots for current experiment
./experiments/xps.py plot

# Generate plots for a specific experiment
./experiments/xps.py plot my_experiment

# Generate plots only for nearfar variants
./experiments/xps.py plot --variant nearfar
```

### `xps.py update [xp_name]`

Rebase the experiment branch on `main` and drop any previous result commits. This is useful for cleaning up a branch before a fresh run or after fixing code issues.

**Example:**
```bash
./experiments/xps.py update
```

## Files

- `xps.py` - Main experiment management tool
- `run_experiments.py` - Old standalone runner (deprecated, use `xps.py run`)
- `experiment_*.cpp` - Experiment runner executables
- `results/` - Generated experiment results (git-ignored except after committed by `xps.py run`)
- `evaluation/` - Analysis and validation tools
- `plots/` - Plotting utilities

## Tips

- Experiment names should use lowercase and underscores only
- Each instrumentation commit is run independently (code changes before it are included)
- Results are automatically committed after a successful run
- Always use `data=berlin` or whatever dataset you have preprocessed in `cache/`
- For meaningful performance comparison, experiments are always built in Release mode
