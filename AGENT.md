# GPUSSSP

GPU-accelerated Single-Source Shortest Path using Vulkan compute shaders.

## Technologies

- **Language**: C++20
- **GPU**: Vulkan (compute shaders, SPIR-V)
- **Build**: CMake 3.20+
- **Testing**: Catch2 v3.5.2
- **Dependencies**: Vulkan SDK, ZLIB, BZip2, libosmium (third_party)

## Development

If you add new helper functions, create the simplest test possible to verify their functionality.
In general avoid comments. Only add them when absolutely neccessary.
We do not need documentation for our function and classes.
Always reference `std` functions with `std::` and don't use `using namespace std;`, same for Vulkan's `vk::` or other third-party libraries.
In our own code it is permissable to do `using namespace gpusssp` in the executable files in `src/`.
After creating sub-tasks always create a git commit. Use one-line git commit messages unless additional context is absolutely neccessary.

## Project Structure

```
include/
  common/      # Core utilities (graphs, coordinates, I/O)
  gpu/         # Vulkan/GPU implementations
  preprocessing/ # OSM data to internal format
src/
  gpusssp.cpp       # Main executable
  osm2graph.cpp     # OSM to graph converter
  delta_step.comp   # Compute shader (GLSL)
tests/              # Catch2 tests
third_party/libosmium/
```

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Testing

Tests are in `tests/`. Run with:

```bash
cd build
ctest
# or
./gpusssp_tests
```

## Formatting

```bash
cmake --build build --target format
```

## Usage

Prepare graph data:

```bash
cd build
./osm2graph ../data/berlin.osm.pbf ../cache/berlin
```

Run E2E test:

```bash
./gpusssp ../cache/berlin 13.263693,52.444990 13.487614,52.580555 300 1
```

Format: `./gpusssp <graph_path> <src_lon,src_lat> <dst_lon,dst_lat> <delta> <num_queries>`
