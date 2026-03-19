# Project Overview

GPUSSSP is a GPU-accelerated Single-Source Shortest Path solver using Vulkan compute shaders. It processes OpenStreetMap data and runs delta-stepping shortest path queries on the GPU.

## Tech Stack

- **Language**: C++20
- **GPU**: Vulkan (compute shaders, SPIR-V)
- **Build**: CMake 3.25+
- **Testing**: Catch2 v3.5.2
- **Dependencies**: Vulkan SDK, ZLIB, BZip2, Threads, glfw3, argparse, libosmium (third_party), stb (third_party)

## Repository Structure

```
include/
  common/           # Core utilities (graphs, coordinates, I/O)
  gpu/              # Vulkan/GPU implementations
  preprocessing/    # OSM data to internal format conversion
src/                # Source files for key executables
shaders/            # GLSL compute shaders
experiments/        # Source files for experiment runners
tests/              # Catch2 unit tests
docs/               # Documentation (naming conventions, etc.)
third_party/
  libosmium/        # OSM file parsing library
  stb/              # Image writing library
data/               # Input OSM data files (.osm.pbf format)
cache/              # Preprocessed graph data (binary format)
```

## Key Files

### Executables
- `src/gpusssp.cpp` - Main entry point for running shortest path queries (compares all algorithms)
- `src/osm2graph.cpp` - Converter from OSM PBF format to internal graph format
- `src/graph_reorder.cpp` - Graph node reordering utility
- `src/visualize_graph.cpp` - Graph visualization using Vulkan + glfw
- `experiments/generate_queries.cpp` - Generate query pairs for experiments
- `experiments/experiment_dijkstra.cpp` - CPU Dijkstra benchmarks
- `experiments/experiment_dial.cpp` - CPU Dial benchmarks
- `experiments/experiment_deltastep.cpp` - GPU DeltaStep benchmarks
- `experiments/experiment_bellmanford.cpp` - GPU BellmanFord benchmarks
- `experiments/experiment_nearfar.cpp` - GPU NearFar benchmarks

### Shaders
- `shaders/delta_step.comp` - Delta-stepping algorithm
- `shaders/deltastep_prepare_dispatch.comp` - Delta-step indirect dispatch setup
- `shaders/bellman_ford.comp` - Bellman-Ford algorithm
- `shaders/nearfar_relax.comp` - Near-Far relaxation
- `shaders/nearfar_compact.comp` - Near-Far bucket compaction
- `shaders/nearfar_prepare_dispatch.comp` - Near-Far indirect dispatch setup
- `shaders/node_color.comp`, `project_coordinates.comp` - Visualization compute shaders
- `shaders/visualize_node.vert`, `visualize_node.frag` - Visualization render shaders
- `shaders/*.glsl` - Shared GLSL include files (`bit_vector.glsl`, `nearfar_constants.glsl`, `statistics.glsl`)

### Core Libraries
- `include/common/` - Core data structures (graphs, coordinates, I/O utilities)
  - `weighted_graph.hpp` - Main graph data structure
  - `dijkstra.hpp` - CPU Dijkstra implementation
  - `dial.hpp` - CPU Dial implementation (bucket-based)
  - `constants.hpp` - Shared constants (INF_WEIGHT, INVALID_ID, etc.)
- `include/gpu/` - Vulkan compute pipeline and GPU management
  - `vulkan_context.hpp` - Vulkan instance and device setup
  - `deltastep.hpp` - DeltaStep GPU algorithm wrapper
  - `bellmanford.hpp` - BellmanFord GPU algorithm wrapper
  - `nearfar.hpp` - NearFar GPU algorithm wrapper
  - `*_buffers.hpp` - GPU memory management for algorithms
- `include/preprocessing/` - OSM data preprocessing utilities

## Algorithms

The project implements five shortest path algorithms:

1. **Dijkstra (CPU)** - Classic CPU implementation used as baseline for correctness
   - Located in `include/common/dijkstra.hpp`
   - Used for validation of GPU algorithms

2. **Dial (CPU)** - Bucket-based shortest path algorithm
   - Located in `include/common/dial.hpp`
   - Uses `BucketQueue` for integer-weighted graphs
   
3. **Delta-Stepping (GPU)** - Parallel shortest path algorithm optimized for GPUs
   - Shader: `shaders/delta_step.comp`
   - Host code: `include/gpu/deltastep.hpp`
   - Splits edges into "light" and "heavy" buckets based on delta parameter
   - Main target algorithm for this project
   
4. **Bellman-Ford (GPU)** - Naive GPU implementation for comparison
   - Shader: `shaders/bellman_ford.comp`
   - Host code: `include/gpu/bellmanford.hpp`
   - Simpler but typically slower than delta-stepping

5. **Near-Far (GPU)** - Advanced GPU algorithm using near/far bucket compaction
   - Shaders: `shaders/nearfar_*.comp`
   - Host code: `include/gpu/nearfar.hpp`
   - Uses indirect dispatch and efficient bucket management for better performance on large graphs

All GPU algorithms use Vulkan compute shaders and process road network graphs from OpenStreetMap data.

## Development Guidelines

### Code Style

#### General
- **Auto when static_cast**: Example `const auto value = static_cast<size_t>(other)`;
- **Prefer default member initialization**: For static values prefer `class s { int value = 0; };`

#### Naming Conventions
The project follows strict naming conventions enforced by clang-tidy. See `docs/NAMING_CONVENTIONS.md` for the full reference.

#### Namespaces
- Always reference `std` functions with `std::` and don't use `using namespace std;`
- Same rule applies for Vulkan's `vk::` and other third-party libraries
- Exception: Using `using namespace gpusssp` is permitted in executable files in `src/`, `experiments/`, and `tests/`
- Use nested namespace declarations: `namespace gpusssp::common` instead of nested blocks
- Close namespaces with comment: `} // namespace gpusssp::common`

#### Header Files
- Use include guards with pattern: `#ifndef GPUSSSP_[PATH]_[FILE]_HPP`
- Example: `include/gpu/deltastep.hpp` → `#ifndef GPUSSSP_GPU_DELTASTEP_HPP`
- Include order: Standard library headers first, then third-party, then project headers
- Always use `#include "path/to/header.hpp"` for project headers (quotes, not angle brackets)

#### Formatting
- Code is formatted using clang-format with the configuration in `.clang-format`
- Run `cmake --build build --target format` to format all code

#### Documentation
- In general avoid comments. Only add them when absolutely necessary, but retain existing comments if still accurate.
- We do not need documentation for functions and classes
- Code should be self-explanatory through good naming

### Testing

- If you add new helper functions, create the simplest test possible to verify their functionality
- Tests are located in `tests/` and use Catch2 v3.5.2
- Run tests with `ctest` or `./gpusssp_tests` from the build directory
- Tests depend on compiled shaders (automatic via CMake dependencies)

### Git Workflow

- After completing sub-tasks always create a git commit
- Use one-line git commit messages unless additional context is absolutely necessary
- Commit message style should match existing commits (check `git log`)
- Don't force-commit files ignored by `.gitignore`

## Common Tasks

### Building

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Running Tests

```bash
cd build
ctest
# or run test binary directly
./gpusssp_tests
```

### Code Formatting

```bash
cmake --build build --target format
```

### Preprocessing OSM Data

```bash
cd build
./osm2graph ../data/berlin.osm.pbf ../cache/berlin
```

This converts OSM PBF format to the internal graph format used by the solver.

### Running Shortest Path Queries

```bash
./gpusssp ../cache/berlin
./gpusssp ../cache/berlin --source 13.38,52.51 --target random --delta 300 --num-queries 10
./gpusssp ../cache/berlin --skip bellmanford,dial
```

The executable uses argparse with optional flags. Run `./gpusssp --help` for full options.

Runs all algorithms (Dijkstra, Dial, DeltaStep, BellmanFord, NearFar) and compares results.

### Running Experiments

The project includes a unified experiment management tool `experiments/xps.py` for systematic benchmarking:
For detailed workflow and examples, see `experiments/README.md`.
