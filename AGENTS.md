# Project Overview

GPUSSSP is a GPU-accelerated Single-Source Shortest Path solver using Vulkan compute shaders. It processes OpenStreetMap data and runs delta-stepping shortest path queries on the GPU.

## Tech Stack

- **Language**: C++20
- **GPU**: Vulkan (compute shaders, SPIR-V)
- **Build**: CMake 3.20+
- **Testing**: Catch2 v3.5.2
- **Dependencies**: Vulkan SDK, ZLIB, BZip2, libosmium (third_party)

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
third_party/
  libosmium/        # OSM file parsing library
data/               # Input OSM data files (.osm.pbf format)
cache/              # Preprocessed graph data (binary format)
.cache/             # Git-ignored cache directory
```

### File Extensions
- `.hpp` - C++ header files (header-only implementations)
- `.cpp` - C++ source files (executables, tests)
- `.comp` - GLSL compute shaders (compiled to .spv)
- `.spv` - SPIR-V compiled shaders (generated in build directory)

## Key Files

### Executables
- `src/gpusssp.cpp` - Main entry point for running shortest path queries (compares Dijkstra, DeltaStep, BellmanFord, NearFar)
- `src/osm2graph.cpp` - Converter from OSM PBF format to internal graph format
- `experiments/generate_queries.cpp` - Generate query pairs for experiments
- `experiments/experiment_dijkstra.cpp` - CPU Dijkstra benchmarks
- `experiments/experiment_deltastep.cpp` - GPU DeltaStep benchmarks
- `experiments/experiment_bellmanford.cpp` - GPU BellmanFord benchmarks
- `experiments/experiment_nearfar.cpp` - GPU NearFar benchmarks

### Shaders
- `shaders/delta_step.comp` - GLSL compute shader implementing delta-stepping algorithm
- `shaders/bellman_ford.comp` - GLSL compute shader implementing Bellman-Ford algorithm
- `shaders/nearfar_relax.comp` - GLSL compute shader for Near-Far relaxation
- `shaders/nearfar_compact.comp` - GLSL compute shader for Near-Far bucket compaction
- `shaders/nearfar_prepare_dispatch.comp` - GLSL compute shader for indirect dispatch setup

### Core Libraries
- `include/common/` - Core data structures (graphs, coordinates, I/O utilities)
  - `weighted_graph.hpp` - Main graph data structure
  - `dijkstra.hpp` - CPU Dijkstra implementation
  - `constants.hpp` - Shared constants (INF_WEIGHT, INVALID_ID, etc.)
- `include/gpu/` - Vulkan compute pipeline and GPU management
  - `vulkan_context.hpp` - Vulkan instance and device setup
  - `deltastep.hpp` - DeltaStep GPU algorithm wrapper
  - `bellmanford.hpp` - BellmanFord GPU algorithm wrapper
  - `nearfar.hpp` - NearFar GPU algorithm wrapper
  - `*_buffers.hpp` - GPU memory management for algorithms
- `include/preprocessing/` - OSM data preprocessing utilities

## Algorithms

The project implements four shortest path algorithms:

1. **Dijkstra (CPU)** - Classic CPU implementation used as baseline for correctness
   - Located in `include/common/dijkstra.hpp`
   - Used for validation of GPU algorithms
   
2. **Delta-Stepping (GPU)** - Parallel shortest path algorithm optimized for GPUs
   - Shader: `shaders/delta_step.comp`
   - Host code: `include/gpu/deltastep.hpp`
   - Splits edges into "light" and "heavy" buckets based on delta parameter
   - Main target algorithm for this project
   
3. **Bellman-Ford (GPU)** - Naive GPU implementation for comparison
   - Shader: `shaders/bellman_ford.comp`
   - Host code: `include/gpu/bellmanford.hpp`
   - Simpler but typically slower than delta-stepping

4. **Near-Far (GPU)** - Advanced GPU algorithm using near/far bucket compaction
   - Shaders: `shaders/nearfar_*.comp`
   - Host code: `include/gpu/nearfar.hpp`
   - Uses indirect dispatch and efficient bucket management for better performance on large graphs

All GPU algorithms use Vulkan compute shaders and process road network graphs from OpenStreetMap data.

## Development Guidelines

### Code Style

#### Naming Conventions
The project follows strict naming conventions enforced by clang-tidy:
- **Classes/Structs/Enums**: `PascalCase` (e.g., `WeightedGraph`, `VulkanContext`)
- **Member Variables**: `snake_case` (e.g., `first_edges`, `heap_size`)
- **Functions/Methods**: `snake_case` (e.g., `num_nodes()`, `create_instance()`)
- **Type Aliases**: `snake_case_t` (e.g., `node_id_t`, `weight_t`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `INVALID_ID`, `INF_WEIGHT`)
- **Namespaces**: `snake_case` (e.g., `gpusssp`, `common`, `gpu`)
- **Template Parameters**: `PascalCaseT` (e.g., `ElementT`, `GraphT`)
- **Enum Values**: `SCREAMING_SNAKE_CASE` (e.g., `DEBUG`, `QUEUE_POP`)

See `docs/NAMING_CONVENTIONS.md` for detailed examples and `docs/CLANG_TIDY.md` for enforcement.

#### Namespaces
- Always reference `std` functions with `std::` and don't use `using namespace std;`
- Same rule applies for Vulkan's `vk::` and other third-party libraries
- Exception: Using `using namespace gpusssp` is permitted in executable files in `src/` and `experiments/`
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
- Key style points from .clang-format:
 - Allman brace style (braces on new lines)
 - 4-space indentation
 - 100 character line limit
 - No bin-packing of arguments/parameters (one per line for readability)

#### Static Analysis
- Code quality is checked using clang-tidy with the configuration in `.clang-tidy`
- Run `cmake --build build --target tidy` to check code
- Run `cmake --build build --target tidy-fix` to apply automatic fixes
- Enable during build: `cmake -DENABLE_CLANG_TIDY=ON -B build`

#### Documentation
- In general avoid comments. Only add them when absolutely necessary
- We do not need documentation for functions and classes
- Code should be self-explanatory through good naming

### Testing

- If you add new helper functions, create the simplest test possible to verify their functionality
- Tests are located in `tests/` and use Catch2 v3.5.2
- Test files use `gpusssp::` prefix (no `using namespace` in tests)
- Run tests with `ctest` or `./gpusssp_tests` from the build directory
- Tests depend on compiled shaders (automatic via CMake dependencies)

### Git Workflow

- After completing sub-tasks always create a git commit
- Use one-line git commit messages unless additional context is absolutely necessary
- Commit message style should match existing commits (check `git log`)
- Don't commit generated files:
  - `build/` directory (compiled binaries, .spv shaders)
  - `.cache/` directory (preprocessed data)
  - Jupyter notebook checkpoints

### Build System

- CMake 3.20+ is required
- Build outputs go to `build/` directory
- Shaders are automatically compiled during build via custom CMake commands
- The build system creates several executables and test binaries
- Use `CMAKE_BUILD_TYPE=Release` for performance benchmarks

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
./gpusssp ../cache/berlin random random 300 1
```

Format: `./gpusssp <graph_path> [<src_lon,src_lat>|random] [<dst_lon,dst_lat>|random] <delta> <num_queries>`

Parameters:
- `graph_path` - Path to preprocessed graph data (without extension)
- `src_lon,src_lat` - Source coordinates (longitude, latitude)
- `dst_lon,dst_lat` - Destination coordinates (longitude, latitude)
- `random` - Use "random" instead of coordinates to pick random nodes
- `delta` - Delta parameter for delta-stepping algorithm (typically 300-3600)
- `num_queries` - Number of queries to run

The executable runs all four algorithms (Dijkstra, DeltaStep, BellmanFord, NearFar) and compares results.

### Running Experiments

The project includes a unified experiment management tool `experiments/xps.py` for systematic benchmarking:

**Create a new experiment:**
```bash
./experiments/xps.py create my_experiment
```
CAUTION: This changes the current branch to `experiment/my_experiment`.
Verify if there are no staged changes that belong on `main` before doing that.

**Add an instrumentation commit:**
```bash
./experiments/xps.py add "deltastep,nearfar" "delta=900,1800,3600 data=berlin"
```

**Run all experiments:**
```bash
./experiments/xps.py run
```

For detailed workflow and examples, see `experiments/README.md`.

## Data Flow

1. **Input**: OSM PBF files (e.g., `data/berlin.osm.pbf`)
2. **Preprocessing**: `osm2graph` converts OSM to binary graph format
   - Outputs: `<name>.graph`, `<name>.coords`, etc. in cache directory
3. **Query Execution**: `gpusssp` loads preprocessed graph and runs queries
4. **Experiments**: Experiment executables for detailed benchmarking

## Useful Tips

- Graph preprocessing is one-time per dataset - reuse cached graphs
- Delta parameter affects performance: smaller = more buckets but finer-grained parallelism
- Always validate GPU results against CPU Dijkstra implementation
- Use Release build for meaningful performance measurements
- Vulkan validation layers are helpful for debugging GPU code
