#include "common/coordinate.hpp"
#include "common/files.hpp"
#include "common/graph_transform.hpp"
#include "common/logger.hpp"
#include "common/timed_logger.hpp"
#include "common/weighted_graph.hpp"
#include "common/zorder.hpp"

#include <algorithm>
#include <argparse/argparse.hpp>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <numeric>
#include <ostream>
#include <string>
#include <vector>

using namespace gpusssp;

int main(int argc, char **argv)
{
    argparse::ArgumentParser program("graph_reorder", "1.0.0");
    program.add_description("Reorder graph nodes using spatial locality optimization.");

    program.add_argument("method")
        .default_value(std::string("zorder"))
        .help("reordering method: zorder");

    program.add_argument("input_path").help("path to input graph data (without extension)");
    program.add_argument("output_path").help("path to output graph data (without extension)");

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception &err)
    {
        common::log_error() << err.what() << '\n';
        common::log_error() << program;
        return EXIT_FAILURE;
    }

    const std::string method = program.get("method");
    const std::string input_path = program.get("input_path");
    const std::string output_path = program.get("output_path");

    if (method != "zorder")
    {
        common::log_error() << "Error: Unknown method '" << method << "'" << '\n';
        common::log_error() << "Available methods: zorder" << '\n';
        return EXIT_FAILURE;
    }

    // Load graph and coordinates
    common::TimedLogger time_read("Loading graph and coordinates");
    auto graph = common::files::read_weighted_graph<std::uint32_t>(input_path);
    auto coordinates = common::files::read_coordinates(input_path);
    time_read.finished();

    if (graph.num_nodes() != coordinates.size())
    {
        common::log() << "Error: Graph has " << graph.num_nodes() << " nodes but "
                      << coordinates.size() << " coordinates" << '\n';
        return EXIT_FAILURE;
    }

    common::log() << "Graph has " << graph.num_nodes() << " nodes and " << graph.num_edges()
                  << " edges" << '\n';

    // Compute z-order values for all nodes
    common::TimedLogger time_zorder("Computing z-order values");
    std::vector<std::uint64_t> zorder_values(graph.num_nodes());
    for (std::uint32_t node = 0; node < graph.num_nodes(); ++node)
    {
        zorder_values[node] = common::coordinate_to_zorder(coordinates[node]);
    }
    time_zorder.finished();

    // Create permutation array: permutation[old_id] = new_id
    // We create an array of indices, then sort it by z-order values
    common::TimedLogger time_permutation("Creating permutation");
    std::vector<std::uint32_t> sorted_nodes(graph.num_nodes());
    std::iota(sorted_nodes.begin(), sorted_nodes.end(), 0);

    // Sort nodes by their z-order values
    std::ranges::sort(sorted_nodes,

                      [&zorder_values](std::uint32_t lhs, std::uint32_t rhs)
                      { return zorder_values[lhs] < zorder_values[rhs]; });

    // Create inverse permutation: old_id -> new_id
    std::vector<std::uint32_t> permutation(graph.num_nodes());
    for (std::uint32_t new_id = 0; new_id < sorted_nodes.size(); ++new_id)
    {
        std::uint32_t old_id = sorted_nodes[new_id];
        permutation[old_id] = new_id;
    }
    time_permutation.finished();

    // Apply permutation to graph edges
    common::TimedLogger time_reorder("Reordering graph");
    auto edges = graph.edges();
    common::renumber_edges(edges, permutation);
    std::ranges::sort(edges);
    auto reordered_graph = common::WeightedGraph<std::uint32_t>(graph.num_nodes(), edges);
    time_reorder.finished();

    // Reorder coordinates
    common::TimedLogger time_coords("Reordering coordinates");
    std::vector<common::Coordinate> reordered_coordinates(coordinates.size());
    for (std::uint32_t old_id = 0; old_id < coordinates.size(); ++old_id)
    {
        std::uint32_t new_id = permutation[old_id];
        reordered_coordinates[new_id] = coordinates[old_id];
    }
    time_coords.finished();

    // Write reordered graph and coordinates
    common::TimedLogger time_write("Writing reordered graph");
    common::files::write_weighted_graph(output_path, reordered_graph);
    common::files::write_coordinates(output_path, reordered_coordinates);
    time_write.finished();

    common::log() << "Successfully reordered graph using " << method << " method" << '\n';

    return EXIT_SUCCESS;
}
