# GPUSSSP Naming Conventions

Quick reference for naming conventions enforced by clang-tidy.

## Summary Table

| Element | Convention | Example |
|---------|-----------|---------|
| **Classes** | `PascalCase` | `WeightedGraph` |
| **Structs** | `PascalCase` | `IDKeyPair` |
| **Enums** | `PascalCase` | `LogLevel` |
| **Member Variables** | `snake_case` | `first_edges` |
| **Functions** | `snake_case` | `num_nodes()` |
| **Methods** | `snake_case` | `create_instance()` |
| **Type Aliases** | `snake_case_t` | `node_id_t` |
| **Constants** | `SCREAMING_SNAKE_CASE` | `INVALID_ID` |
| **Namespaces** | `snake_case` | `gpusssp::common` |
| **Template Params** | `PascalCaseT` | `ElementT`, `GraphT` |
| **Enum Values** | `SCREAMING_SNAKE_CASE` | `DEBUG`, `INFO` |
| **Parameters** | `snake_case` | `src_node` |
| **Local Variables** | `snake_case` | `best_cost` |

## Detailed Examples

### Classes and Structs

```cpp
// ✅ Correct
class WeightedGraph { };
struct IDKeyPair { };
enum class LogLevel { };

// ❌ Wrong
class weighted_graph { };      // Should be PascalCase
struct id_key_pair { };        // Should be PascalCase
enum class log_level { };      // Should be PascalCase
```

### Member Variables

```cpp
class MyClass
{
    // ✅ Correct
    std::vector<int> first_edges;
    std::size_t heap_size;
    
    // ❌ Wrong
    std::vector<int> FirstEdges;     // Should be snake_case
    std::size_t m_heap_size;         // No prefix allowed
    std::size_t _heap_size;          // No prefix allowed
};
```

### Functions and Methods

```cpp
// ✅ Correct
std::size_t num_nodes() const;
void create_instance();
void alloc_and_bind();

// ❌ Wrong
std::size_t NumNodes() const;        // Should be snake_case
void CreateInstance();               // Should be snake_case
void allocAndBind();                 // Should be snake_case
```

### Type Aliases

```cpp
// ✅ Correct
using node_id_t = std::uint32_t;
using weight_t = std::int32_t;
using edge_t = Edge<node_id_t, weight_t>;

// ❌ Wrong
using NodeId = std::uint32_t;       // Should be snake_case with _t
using Weight = std::int32_t;        // Missing _t suffix
using edge = Edge<...>;             // Missing _t suffix
```

### Constants

```cpp
// ✅ Correct
constexpr std::uint32_t INVALID_ID = std::numeric_limits<std::uint32_t>::max();
constexpr std::int32_t INF_WEIGHT = std::numeric_limits<std::int32_t>::max() / 2;
constexpr double FIXED_POINT_RESOLUTION = 10.0;

// ❌ Wrong
constexpr std::uint32_t invalid_id = ...;    // Should be SCREAMING_SNAKE_CASE
constexpr std::int32_t InfWeight = ...;      // Should be SCREAMING_SNAKE_CASE
const double FixedPointResolution = ...;     // Should be SCREAMING_SNAKE_CASE
```

### Namespaces

```cpp
// ✅ Correct
namespace gpusssp {
namespace common {
    // ...
} // namespace common
} // namespace gpusssp

// Or nested:
namespace gpusssp::common {
    // ...
} // namespace gpusssp::common

// ❌ Wrong
namespace Gpusssp {          // Should be all lowercase
namespace COMMON {           // Should be all lowercase
```

### Template Parameters

```cpp
// ✅ Correct
template <typename ElementT>
class LazyClearVector { };

template <typename GraphT>
class DeltaStep { };

template <typename T>        // OK for single generic parameter
void process(T value);

// ❌ Wrong
template <typename Element>  // Should have T suffix
class LazyClearVector { };

template <typename graph_t>  // Should be PascalCase
class DeltaStep { };
```

### Enum Values

```cpp
// ✅ Correct
enum class LogLevel
{
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

enum class StatisticsEvent
{
    QUEUE_POP,
    DIJKSTRA_RELAX,
    NUM_EVENTS
};

// ❌ Wrong
enum class LogLevel
{
    Debug,        // Should be SCREAMING_SNAKE_CASE
    Info,         // Should be SCREAMING_SNAKE_CASE
};
```

### Function Parameters

```cpp
// ✅ Correct
void process(uint32_t src_node, uint32_t dst_node);

// Constructor with shadowing parameters
class MyClass
{
    MyClass(std::vector<int> first_edges_,
            std::vector<int> targets_)
        : first_edges(std::move(first_edges_)),
          targets(std::move(targets_))
    { }
    
    std::vector<int> first_edges;
    std::vector<int> targets;
};

// ❌ Wrong
void process(uint32_t SrcNode, uint32_t DstNode);    // Should be snake_case
void process(uint32_t srcNode, uint32_t dstNode);    // Should be snake_case
```

### Local Variables

```cpp
void my_function()
{
    // ✅ Correct
    auto num_nodes = graph.num_nodes();
    auto best_cost = INF_WEIGHT;
    auto tentative_cost = cost + weight;
    
    // ❌ Wrong
    auto NumNodes = graph.num_nodes();          // Should be snake_case
    auto BestCost = INF_WEIGHT;                 // Should be snake_case
    auto tentativeCost = cost + weight;         // Should be snake_case
}
```

## Special Cases

### Abbreviations

Treat abbreviations as words:

```cpp
// ✅ Correct
class IdQueue { };              // ID treated as word "Id"
struct IdKeyPair { };
using node_id_t = uint32_t;     // lowercase in snake_case

// ❌ Wrong (but currently exists in codebase - needs fixing)
struct IDKeyPair { };           // Should be IdKeyPair
```

### Vulkan Types

When interfacing with Vulkan, use Vulkan's naming for their types:

```cpp
// ✅ Correct
vk::Device device;              // Vulkan types keep their naming
vk::Buffer buf_dist;            // Our variable uses snake_case
vk::CommandBuffer cmd_buf;      // Our variable uses snake_case
```

### Lambda Parameters

Follow same rules as function parameters:

```cpp
// ✅ Correct
auto lambda = [](const auto& lhs, const auto& rhs) {
    return lhs.weight < rhs.weight;
};

// ❌ Wrong
auto lambda = [](const auto& Lhs, const auto& Rhs) {  // Should be snake_case
    return Lhs.weight < Rhs.weight;
};
```

## Migration Guide

If you have existing code that doesn't follow these conventions:

1. **Run clang-tidy to identify issues:**
   ```bash
   cmake --build build --target tidy
   ```

2. **Apply automatic fixes (carefully):**
   ```bash
   cmake --build build --target tidy-fix
   ```

3. **Review changes before committing:**
   ```bash
   git diff
   ```

4. **Fix any remaining issues manually**

## Enforcement

These conventions are enforced by:
- **clang-tidy**: Static analysis during build or via `make tidy`
- **Code review**: Human review of pull requests
- **CI/CD**: Automated checks in continuous integration

To enable automatic enforcement during build:
```bash
cmake -DENABLE_CLANG_TIDY=ON -B build
cmake --build build
```

## Questions?

See `docs/CLANG_TIDY.md` for detailed information about the clang-tidy integration.
