#include "preprocessing/import_osm.hpp"

#include "common/files.hpp"
#include "common/logger.hpp"
#include "common/timed_logger.hpp"
#include "common/weighted_graph.hpp"

using namespace gpusssp;

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        common::log() << argv[0] << " PATH_TO_OSM OUTPUT_PATH" << '\n';
        common::log() << "Example: " << argv[0] << " berlin.osm.pbf cache/graph/berlin" << '\n';
        return EXIT_FAILURE;
    }

    const std::string network_path = argv[1];
    const std::string output_base_path = argv[2];

    common::TimedLogger time_read("Reading network from");
    auto network = preprocessing::read_network(network_path);
    time_read.finished();

    common::TimedLogger time_simplify("Simplify OSM data");
    auto nodes_before = network.nodes.size();
    network = preprocessing::simplify_network(std::move(network));
    auto nodes_after = network.nodes.size();
    time_simplify.finished();
    common::log() << "Removed " << (nodes_before - nodes_after) << "("
                  << ((nodes_before - nodes_after) / (double)nodes_before * 100) << "%) nodes."
                  << '\n';

    common::TimedLogger time_convert("Converting to graph");
    auto graph = preprocessing::weighted_graph_from_network(network);
    auto coordinates = preprocessing::coordinates_from_network(network);
    time_convert.finished();

    common::log() << "Graph has " << graph.num_nodes() << " nodes and " << graph.num_edges()
                  << " edges" << '\n';

    common::TimedLogger time_write("Writing graph");
    common::files::write_weighted_graph(output_base_path, graph);
    common::files::write_coordinates(output_base_path, coordinates);
    time_write.finished();

    return EXIT_SUCCESS;
}
