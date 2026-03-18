#include <algorithm>
#include <argparse/argparse.hpp>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
#include <vulkan/vulkan.hpp>

#include "common/constants.hpp"
#include "common/dijkstra.hpp"
#include "common/files.hpp"
#include "common/id_queue.hpp"
#include "common/logger.hpp"
#include "common/path.hpp"
#include "common/weighted_graph.hpp"

#include "gpu/debug.hpp"
#include "gpu/graph_buffers.hpp"
#include "gpu/nearfar.hpp"
#include "gpu/nearfar_buffers.hpp"
#include "gpu/statistics.hpp"
#include "gpu/vulkan_context.hpp"

using namespace gpusssp;

namespace
{

auto extract_subgraph(const common::WeightedGraph<uint32_t> &graph,
                      const std::vector<bool> &node_filter)
{
    std::vector<uint32_t> old_to_new(graph.num_nodes(), common::INVALID_ID);
    std::vector<uint32_t> filtered_nodes;
    for (uint32_t old_id = 0; old_id < graph.num_nodes(); ++old_id)
    {
        if (node_filter[old_id])
        {
            old_to_new[old_id] = static_cast<uint32_t>(filtered_nodes.size());
            filtered_nodes.push_back(old_id);
        }
    }

    std::vector<common::Edge<uint32_t, uint32_t>> subgraph_edges;
    for (uint32_t old_id : filtered_nodes)
    {
        for (auto eid = graph.begin(old_id); eid < graph.end(old_id); ++eid)
        {
            auto target = graph.target(eid);
            if (node_filter[target])
            {
                subgraph_edges.emplace_back(
                    old_to_new[old_id], old_to_new[target], graph.weight(eid));
            }
        }
    }

    std::sort(subgraph_edges.begin(),
              subgraph_edges.end(),
              [](const auto &lhs, const auto &rhs)
              { return std::tie(lhs.start, lhs.target) < std::tie(rhs.start, rhs.target); });

    common::WeightedGraph<uint32_t> subgraph(filtered_nodes.size(), subgraph_edges);

    return std::make_tuple(std::move(subgraph), std::move(old_to_new));
}
} // namespace

int main(int argc, char **argv)
{
    argparse::ArgumentParser program("extract_minimal_graph", "1.0.0");
    program.add_description("Extract minimal subgraph that reproduces NearFar algorithm bugs.");

    program.add_argument("graph_path").help("path to preprocessed graph data (without extension)");
    program.add_argument("source_node").scan<'u', uint32_t>().help("source node ID");
    program.add_argument("target_node").scan<'u', uint32_t>().help("target node ID");
    program.add_argument("delta").scan<'u', uint32_t>().help("delta parameter for NearFar");
    program.add_argument("output_path").help("output directory for minimal graph");

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception &err)
    {
        std::cerr << err.what() << '\n';
        std::cerr << program;
        return 1;
    }

    std::string graph_path = program.get("graph_path");
    auto source_node = program.get<uint32_t>("source_node");
    auto target_node = program.get<uint32_t>("target_node");
    auto delta = program.get<uint32_t>("delta");
    std::string output_path = program.get("output_path");

    common::log() << "Loading graph from: " << graph_path << '\n';
    auto graph = common::files::read_weighted_graph<uint32_t>(graph_path);
    common::log() << "Graph loaded: " << graph.num_nodes() << " nodes, " << graph.num_edges()
                  << " edges\n";

    if (source_node >= graph.num_nodes())
    {
        common::log_error() << "Error: source_node " << source_node << " is out of range\n";
        return 1;
    }
    if (target_node >= graph.num_nodes())
    {
        common::log_error() << "Error: target_node " << target_node << " is out of range\n";
        return 1;
    }

    common::MinIDQueue min_queue(graph.num_nodes());
    common::CostVector<common::WeightedGraph<uint32_t>> dijkstra_costs(graph.num_nodes(),
                                                                       common::INF_WEIGHT);
    common::ParentVector<common::WeightedGraph<uint32_t>> parents(graph.num_nodes(),
                                                                  common::INVALID_ID);

    common::log() << "Running Dijkstra from " << source_node << " to " << target_node << '\n';
    auto dijkstra_dist =
        common::dijkstra(source_node, target_node, graph, min_queue, dijkstra_costs, parents);

    if (dijkstra_dist == common::INF_WEIGHT)
    {
        common::log_error() << "Error: Target node " << target_node << " is unreachable from "
                            << source_node << '\n';
        return 1;
    }

    common::log() << "Dijkstra distance: " << dijkstra_dist << '\n';

    gpu::VulkanContext vk_ctx("ExtractMinimalGraph", gpu::detail::select_device());
    auto device = vk_ctx.device();
    auto queue = vk_ctx.queue();
    auto cmd_pool = vk_ctx.command_pool();

    gpu::Statistics gpu_statistics(device, vk_ctx.memory_properties());
    gpu::GraphBuffers graph_buffers(graph, device, vk_ctx.memory_properties(), cmd_pool, queue);
    gpu::NearFarBuffers nearfar_buffers(graph.num_nodes(), device, vk_ctx.memory_properties());

    gpu::NearFar nearfar(graph_buffers, nearfar_buffers, device, gpu_statistics, delta);
    nearfar.initialize(cmd_pool);

    auto nearfar_dist = nearfar.run(cmd_pool, queue, source_node, target_node);

    if (nearfar_dist == dijkstra_dist)
    {
        common::log() << "No divergence found: NearFar and Dijkstra agree.\n";
        return 0;
    }

    common::log() << "Mismatch detected! dijkstra " << dijkstra_dist << " != nearfar "
                  << nearfar_dist << '\n';

    auto nearfar_distances = gpu::read_buffer<uint32_t>(device,
                                                        vk_ctx.memory_properties(),
                                                        cmd_pool,
                                                        queue,
                                                        nearfar_buffers.dist_buffer(),
                                                        graph.num_nodes());

    auto shortest_path =
        common::get_path<common::WeightedGraph<uint32_t>>(source_node, target_node, parents);

    uint32_t first_divergent_node = common::INVALID_ID;
    for (auto node : shortest_path)
    {
        if (dijkstra_costs[node] != nearfar_distances[node])
        {
            first_divergent_node = node;
            break;
        }
    }

    if (first_divergent_node == common::INVALID_ID)
    {
        common::log() << "No divergence found on shortest path.\n";
        return 0;
    }

    common::log() << "First divergent node: " << first_divergent_node << '\n';
    common::log() << "  Dijkstra distance: " << dijkstra_costs[first_divergent_node] << '\n';
    common::log() << "  NearFar distance: " << nearfar_distances[first_divergent_node] << '\n';

    if (first_divergent_node == source_node)
    {
        common::log_error()
            << "Error: First divergent node is the source. This shouldn't happen.\n";
        return 1;
    }

    common::MinIDQueue settle_queue(graph.num_nodes());
    common::CostVector<common::WeightedGraph<uint32_t>> settle_costs(graph.num_nodes(),
                                                                     common::INF_WEIGHT);
    std::vector<bool> settled(graph.num_nodes(), false);

    common::log() << "Extracting settled nodes from " << source_node << " to "
                  << first_divergent_node << '\n';
    common::dijkstra(source_node, first_divergent_node, graph, settle_queue, settle_costs, settled);

    auto [subgraph, old_to_new] = extract_subgraph(graph, settled);

    common::log() << "Subgraph created: " << subgraph.num_nodes() << " nodes, "
                  << subgraph.num_edges() << " edges\n"
                  << "\tsource: " << old_to_new[source_node] << '\n'
                  << "\ttarget: " << old_to_new[first_divergent_node] << '\n';

    std::filesystem::create_directories(output_path);
    common::files::write_weighted_graph(output_path, subgraph);
    common::log() << "Written to: " << output_path << '\n';

    return 0;
}
