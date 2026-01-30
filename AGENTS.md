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
src/                # Source files for key executables and shader
experiments/        # Source files for experiment runners
tests/              # Catch2 unit tests
third_party/
  libosmium/        # OSM file parsing library
data/               # Input OSM data files
cache/              # Preprocessed graph data
```

## Key Files

- `src/delta_step.comp` - GLSL compute shader implementing delta-stepping algorithm
- `src/gpusssp.cpp` - Main entry point for running shortest path queries
- `src/osm2graph.cpp` - Converter from OSM PBF format to internal graph format
- `include/common/` - Core data structures (graphs, coordinates, I/O utilities)
- `include/gpu/` - Vulkan compute pipeline and GPU management
- `include/preprocessing/` - OSM data preprocessing utilities

## Development Guidelines

### Code Style

- Always reference `std` functions with `std::` and don't use `using namespace std;`
- Same rule applies for Vulkan's `vk::` and other third-party libraries
- Exception: Using `using namespace gpusssp` is permitted in executable files in `src/`
- In general avoid comments. Only add them when absolutely necessary
- We do not need documentation for functions and classes

### Testing

- If you add new helper functions, create the simplest test possible to verify their functionality
- Tests are located in `tests/` and use Catch2 v3.5.2

### Git Workflow

- After creating sub-tasks always create a git commit
- Use one-line git commit messages unless additional context is absolutely necessary

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
- `graph_path` - Path to preprocessed graph data
- `src_lon,src_lat` - Source coordinates (longitude, latitude)
- `dst_lon,dst_lat` - Destination coordinates (longitude, latitude)
- `random` - Picks a random coordinate
- `delta` - Delta parameter for delta-stepping algorithm
- `num_queries` - Number of queries to run
