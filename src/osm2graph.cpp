#include "preprocessing/import_osm.hpp"

#include "common/files.hpp"
#include "common/logger.hpp"
#include "common/timed_logger.hpp"
#include "common/weighted_graph.hpp"

#include <argparse/argparse.hpp>
#include <cstdlib>
#include <exception>
#include <string>

using namespace gpusssp;

int main(int argc, char **argv)
{
    argparse::ArgumentParser program("osm2graph", "1.0.0");
    program.add_description("Convert OSM PBF format to internal graph format.");

    program.add_argument("osm_path").help("path to OSM PBF file");
    program.add_argument("output_path").help("output path for graph data (without extension)");

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

    const std::string network_path = program.get("osm_path");
    const std::string output_base_path = program.get("output_path");

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
