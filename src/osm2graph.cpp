#include "preprocessing/import_osm.hpp"

#include "common/files.hpp"
#include "common/timed_logger.hpp"
#include "common/weighted_graph.hpp"

#include <iostream>

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << argv[0] << " PATH_TO_OSM OUTPUT_PATH" << std::endl;
        std::cout << "Example: " << argv[0] << " berlin.osm.pbf cache/graph/berlin" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string network_path = argv[1];
    const std::string output_base_path = argv[2];

    gpusssp::common::TimedLogger time_read("Reading network from");
    auto network = gpusssp::preprocessing::read_network(network_path);
    time_read.finished();

    gpusssp::common::TimedLogger time_simplify("Simplify OSM data");
    auto nodes_before = network.nodes.size();
    network = gpusssp::preprocessing::simplify_network(std::move(network));
    auto nodes_after = network.nodes.size();
    time_simplify.finished();
    std::cerr << "Removed " << (nodes_before - nodes_after) << "("
              << ((nodes_before - nodes_after) / (double)nodes_before * 100) << "%) nodes."
              << std::endl;

    gpusssp::common::TimedLogger time_convert("Converting to graph");
    auto graph = gpusssp::preprocessing::weighted_graph_from_network(network);
    auto coordinates = gpusssp::preprocessing::coordinates_from_network(network);
    time_convert.finished();

    std::cerr << "Graph has " << graph.num_nodes() << " nodes and " << graph.num_edges() << " edges"
              << std::endl;

    gpusssp::common::TimedLogger time_write("Writing graph");
    gpusssp::common::files::write_weighted_graph(output_base_path, graph);
    gpusssp::common::files::write_coordinates(output_base_path, coordinates);
    time_write.finished();

    return EXIT_SUCCESS;
}
