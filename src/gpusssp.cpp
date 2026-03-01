#include <algorithm>
#include <cassert>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <ostream>
#include <random>
#include <string>
#include <vector>
#include <vulkan/vulkan.hpp>

#include "common/benchmark.hpp"
#include "common/constants.hpp"
#include "common/coordinate.hpp"
#include "common/dijkstra.hpp"
#include "common/files.hpp"
#include "common/graph_metrics.hpp"
#include "common/id_queue.hpp"
#include "common/logger.hpp"
#include "common/nearest_neighbour.hpp"
#include "common/weighted_graph.hpp"

#include "gpu/bellmanford.hpp"
#include "gpu/bellmanford_buffers.hpp"
#include "gpu/deltastep.hpp"
#include "gpu/deltastep_buffers.hpp"
#include "gpu/device_info.hpp"
#include "gpu/graph_buffers.hpp"
#include "gpu/nearfar.hpp"
#include "gpu/nearfar_buffers.hpp"
#include "gpu/statistics.hpp"
#include "gpu/vulkan_context.hpp"

using namespace gpusssp;

static std::optional<common::Coordinate> string_to_coordinate(const std::string &s)
{
    if (s == "random")
    {
        return {};
    }

    auto pos = s.find(',');
    if (pos == std::string::npos)
    {
        return {};
    }
    return common::Coordinate::from_floating(std::stod(s.substr(0, pos)),
                                             std::stod(s.substr(pos + 1)));
}

static std::optional<uint32_t> string_to_node_id(const std::string &s)
{
    if (s.empty() || !std::isdigit(s[0]))
    {
        return {};
    }

    std::size_t pos;
    auto node_id = std::stoi(s, &pos);
    if (pos != s.size())
    {
        return {};
    }

    return node_id;
}

int main(int argc, char **argv)
{
    // Parse command line arguments
    if (argc < 2 || argc > 6)
    {
        common::log_error()
            << "Usage: " << argv[0]
            << " <graph_path> [SRC_LON,SRC_LAT DEST_LON,DEST_LAT] [DELTA] [NUM_QUERIES]" << '\n';
        common::log_error() << "Example: " << argv[0]
                            << " cache/berlin 13.3889,52.5170 13.4050,52.5200 3600 1" << '\n';
        return 1;
    }

    std::string graph_path = argv[1];
    std::optional<common::Coordinate> maybe_src_coord;
    std::optional<common::Coordinate> maybe_dst_coord;
    if (argc > 2)
    {
        maybe_src_coord = string_to_coordinate(argv[2]);
        maybe_dst_coord = string_to_coordinate(argv[3]);
    }
    std::optional<uint32_t> maybe_src_node_id;
    std::optional<uint32_t> maybe_dst_node_id;
    if (argc > 2)
    {
        if (!maybe_src_coord)
            maybe_src_node_id = string_to_node_id(argv[2]);
        if (!maybe_dst_coord)
            maybe_dst_node_id = string_to_node_id(argv[3]);
    }

    auto num_queries = 1u;
    if (argc >= 6)
    {
        num_queries = std::stoi(argv[5]);
    }

    common::log() << "Loading graph from: " << graph_path << '\n';
    auto graph = common::files::read_weighted_graph<uint32_t>(graph_path);
    auto coordinates = common::files::read_coordinates(graph_path);

    auto delta = common::compute_delta_heuristic(graph);
    if (argc >= 5 && std::string(argv[4]) != "auto")
    {
        delta = std::stoi(argv[4]);
    }
    common::log() << "Using delta value " << delta << '\n';

    auto num_heavy = 0u;
    for (uint32_t eid = 0u; eid < graph.num_edges(); ++eid)
    {
        num_heavy += graph.weight(eid) >= delta;
    }

    common::log() << "Graph loaded: " << graph.num_nodes() << " nodes, " << graph.num_edges()
                  << " edges (" << num_heavy << " heavy)" << '\n';

    common::NearestNeighbour nn(coordinates);

    std::uniform_int_distribution<> random_node_id(0, graph.num_nodes() - 1);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<uint32_t> src_nodes(num_queries);
    std::vector<uint32_t> dst_nodes(num_queries);
    if (maybe_src_coord)
    {
        std::ranges::fill(src_nodes, nn.nearest(*maybe_src_coord));
    }
    else if (maybe_src_node_id)
    {
        std::ranges::fill(src_nodes, *maybe_src_node_id);
    }
    else
    {
        std::ranges::generate(src_nodes, [&]() { return random_node_id(gen); });
    }
    if (maybe_dst_coord)
    {
        std::ranges::fill(dst_nodes, nn.nearest(*maybe_dst_coord));
    }
    else if (maybe_dst_node_id)
    {
        std::ranges::fill(dst_nodes, *maybe_dst_node_id);
    }
    else
    {
        std::ranges::generate(dst_nodes, [&]() { return random_node_id(gen); });
    }

    gpu::VulkanContext vk_ctx("DeltaStep", gpu::detail::select_device());

    gpu::print_device_info(vk_ctx);
    auto device = vk_ctx.device();
    auto queue = vk_ctx.queue();
    auto cmd_pool = vk_ctx.command_pool();

    {
        common::MinIDQueue min_queue(graph.num_nodes());
        common::CostVector<common::WeightedGraph<uint32_t>> costs(graph.num_nodes(),
                                                                  common::INF_WEIGHT);
        std::vector<bool> settled(graph.num_nodes(), false);

        gpu::GraphBuffers graph_buffers(graph, device, vk_ctx.memory_properties(), cmd_pool, queue);
        gpu::DeltaStepBuffers deltastep_buffers(
            graph.num_nodes(), device, vk_ctx.memory_properties());
        gpu::BellmanFordBuffers bellmanford_buffers(
            graph.num_nodes(), device, vk_ctx.memory_properties());
        gpu::NearFarBuffers nearfar_buffers(graph.num_nodes(), device, vk_ctx.memory_properties());
        gpu::Statistics gpu_statistics(device, vk_ctx.memory_properties());

        gpu::DeltaStep deltastep(graph_buffers, deltastep_buffers, device, gpu_statistics);
        deltastep.initialize();

        gpu::BellmanFord bellmanford(graph_buffers, bellmanford_buffers, device, gpu_statistics);
        bellmanford.initialize();

        gpu::NearFar nearfar(graph_buffers, nearfar_buffers, device, gpu_statistics, delta);
        nearfar.initialize();

        std::uint32_t checksum = 0;
        std::uint32_t dij_duration = 0;
        std::uint32_t ds_duration = 0;
        std::uint32_t bf_duration = 0;
        std::uint32_t nf_duration = 0;
        std::size_t num_unreachable = 0;
        for (auto i = 0u; i < num_queries; i++)
        {
            common::log() << "Running " << src_nodes[i] << "->" << dst_nodes[i] << '\n';
            auto time_1 = std::chrono::high_resolution_clock::now();
            auto expected_dist =
                common::dijkstra(src_nodes[i], dst_nodes[i], graph, min_queue, costs, settled);
            common::do_not_optimize(expected_dist);
            auto time_2 = std::chrono::high_resolution_clock::now();
            auto dist = deltastep.run(cmd_pool, queue, src_nodes[i], dst_nodes[i], delta);
            common::do_not_optimize(dist);
            auto time_3 = std::chrono::high_resolution_clock::now();
            auto bf_dist = bellmanford.run(cmd_pool, queue, src_nodes[i], dst_nodes[i]);
            common::do_not_optimize(bf_dist);
            auto time_4 = std::chrono::high_resolution_clock::now();
            auto nf_dist = nearfar.run(cmd_pool, queue, src_nodes[i], dst_nodes[i]);
            common::do_not_optimize(nf_dist);
            auto time_5 = std::chrono::high_resolution_clock::now();

            if (expected_dist == common::INF_WEIGHT)
            {
                num_unreachable++;
                continue;
            }

            dij_duration +=
                std::chrono::duration_cast<std::chrono::milliseconds>(time_2 - time_1).count();
            ds_duration +=
                std::chrono::duration_cast<std::chrono::milliseconds>(time_3 - time_2).count();
            bf_duration +=
                std::chrono::duration_cast<std::chrono::milliseconds>(time_4 - time_3).count();
            nf_duration +=
                std::chrono::duration_cast<std::chrono::milliseconds>(time_5 - time_4).count();
            if (dist != expected_dist)
            {
                common::log_error()
                    << "Error: DeltaStep distance " << src_nodes[i] << "->" << dst_nodes[i]
                    << " mismatch. expected: " << expected_dist << " actual: " << dist << '\n';
            }
            if (bf_dist != expected_dist)
            {
                common::log_error()
                    << "Error: BellmanFord distance " << src_nodes[i] << "->" << dst_nodes[i]
                    << " mismatch. expected: " << expected_dist << " actual: " << bf_dist << '\n';
            }
            if (nf_dist != expected_dist)
            {
                common::log_error()
                    << "Error: NearFar distance " << src_nodes[i] << "->" << dst_nodes[i]
                    << " mismatch. expected: " << expected_dist << " actual: " << nf_dist << '\n';
            }
            checksum += dist;
        }
        auto num_reachable = num_queries - num_unreachable;
        common::log() << "Processed " << num_reachable << " queries (" << num_unreachable
                      << " unreachable) in " << (dij_duration / num_reachable)
                      << "ms/req (dijkstra) " << (ds_duration / num_reachable)
                      << "ms/req (deltastep " << (dij_duration / static_cast<double>(ds_duration))
                      << ") " << (bf_duration / num_reachable) << "ms/req (bellmanford "
                      << (dij_duration / static_cast<double>(bf_duration)) << ") "
                      << (nf_duration / num_reachable) << "ms/req (nearfar "
                      << (dij_duration / static_cast<double>(nf_duration)) << ")" << '\n';
        common::log() << "Checksum: " << (checksum / num_reachable) << '\n';

#ifdef ENABLE_STATISTICS
        common::log() << "Statistics: " << std::endl
                      << common::Statistics::get().summary() << gpu_statistics.summary()
                      << std::endl;
#endif
    }

    return 0;
}
