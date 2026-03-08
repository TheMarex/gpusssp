#include <algorithm>
#include <argparse/argparse.hpp>
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
#include "common/bucket_queue.hpp"
#include "common/constants.hpp"
#include "common/coordinate.hpp"
#include "common/dial.hpp"
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
    argparse::ArgumentParser program("gpusssp", "1.0.0");
    program.add_description("GPU-accelerated Single-Source Shortest Path solver.");

    program.add_argument("graph_path")
        .help("path to preprocessed graph data (without extension)");

    program.add_argument("-s", "--source")
        .default_value(std::string("random"))
        .help("source node: \"random\", node ID, or lon,lat");

    program.add_argument("-t", "--target")
        .default_value(std::string("random"))
        .help("target node: \"random\", node ID, or lon,lat");

    program.add_argument("-d", "--delta")
        .default_value(std::string("auto"))
        .help("delta parameter for delta-stepping: \"auto\" or integer");

    program.add_argument("-n", "--num-queries")
        .default_value(1u)
        .scan<'u', uint32_t>()
        .help("number of queries to run");

    program.add_argument("--skip")
        .default_value(std::string("bellmanford"))
        .help("comma-separated list of algorithms to skip: dijkstra,dial,deltastep,bellmanford,nearfar");

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
    auto source_str = program.get("--source");
    auto target_str = program.get("--target");
    auto delta_str = program.get("--delta");
    auto num_queries = program.get<uint32_t>("--num-queries");

    auto skip_str = program.get("--skip");
    auto is_skipped = [&](const std::string &name)
    {
        std::size_t start = 0;
        while (start < skip_str.size())
        {
            auto end = skip_str.find(',', start);
            if (end == std::string::npos)
                end = skip_str.size();
            if (skip_str.substr(start, end - start) == name)
                return true;
            start = end + 1;
        }
        return false;
    };

    auto maybe_src_coord = string_to_coordinate(source_str);
    auto maybe_dst_coord = string_to_coordinate(target_str);
    auto maybe_src_node_id = maybe_src_coord ? std::nullopt : string_to_node_id(source_str);
    auto maybe_dst_node_id = maybe_dst_coord ? std::nullopt : string_to_node_id(target_str);

    common::log() << "Loading graph from: " << graph_path << '\n';
    auto graph = common::files::read_weighted_graph<uint32_t>(graph_path);
    auto coordinates = common::files::read_coordinates(graph_path);

    auto delta = common::compute_delta_heuristic(graph);
    if (delta_str != "auto")
    {
        delta = std::stoi(delta_str);
    }
    common::log() << "Using delta value " << delta << '\n';

    const auto max_path_weight = common::estimate_max_path_weight(coordinates);
    common::log() << "Using maximum path weight " << max_path_weight << '\n';

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
        common::BucketQueue bucket_queue(graph.num_nodes(), max_path_weight + 1);
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

        gpu::DeltaStep deltastep(graph_buffers, deltastep_buffers, device, gpu_statistics, delta);
        deltastep.initialize(cmd_pool);

        gpu::BellmanFord bellmanford(graph_buffers, bellmanford_buffers, device, gpu_statistics);
        bellmanford.initialize(cmd_pool);

        gpu::NearFar nearfar(graph_buffers, nearfar_buffers, device, gpu_statistics, delta);
        nearfar.initialize(cmd_pool);

        const bool skip_dijkstra = is_skipped("dijkstra");
        const bool skip_dial = is_skipped("dial");
        const bool skip_deltastep = is_skipped("deltastep");
        const bool skip_bellmanford = is_skipped("bellmanford");
        const bool skip_nearfar = is_skipped("nearfar");

        std::uint32_t checksum = 0;
        std::uint32_t dij_duration = 0;
        std::uint32_t dl_duration = 0;
        std::uint32_t ds_duration = 0;
        std::uint32_t bf_duration = 0;
        std::uint32_t nf_duration = 0;
        std::size_t num_unreachable = 0;
        for (auto i = 0u; i < num_queries; i++)
        {
            common::log() << "Running " << src_nodes[i] << "->" << dst_nodes[i] << '\n';

            auto time_dij_start = std::chrono::high_resolution_clock::now();
            auto expected_dist = common::INF_WEIGHT;
            if (!skip_dijkstra)
            {
                expected_dist =
                    common::dijkstra(src_nodes[i], dst_nodes[i], graph, min_queue, costs, settled);
                common::do_not_optimize(expected_dist);
            }
            auto time_dij_end = std::chrono::high_resolution_clock::now();

            auto dist = common::INF_WEIGHT;
            if (!skip_deltastep)
            {
                dist = deltastep.run(cmd_pool, queue, src_nodes[i], dst_nodes[i]);
                common::do_not_optimize(dist);
            }
            auto time_ds_end = std::chrono::high_resolution_clock::now();

            auto bf_dist = common::INF_WEIGHT;
            if (!skip_bellmanford)
            {
                bf_dist = bellmanford.run(cmd_pool, queue, src_nodes[i], dst_nodes[i]);
                common::do_not_optimize(bf_dist);
            }
            auto time_bf_end = std::chrono::high_resolution_clock::now();

            auto nf_dist = common::INF_WEIGHT;
            if (!skip_nearfar)
            {
                nf_dist = nearfar.run(cmd_pool, queue, src_nodes[i], dst_nodes[i]);
                common::do_not_optimize(nf_dist);
            }
            auto time_nf_end = std::chrono::high_resolution_clock::now();

            auto dl_dist = common::INF_WEIGHT;
            if (!skip_dial)
            {
                dl_dist = dial(src_nodes[i], dst_nodes[i], graph, bucket_queue, costs, settled);
                common::do_not_optimize(dl_dist);
            }
            auto time_dl_end = std::chrono::high_resolution_clock::now();

            if (expected_dist == common::INF_WEIGHT && !skip_dijkstra)
            {
                num_unreachable++;
                continue;
            }

            if (!skip_dijkstra)
            {
                dij_duration += std::chrono::duration_cast<std::chrono::milliseconds>(
                                    time_dij_end - time_dij_start)
                                    .count();
            }
            if (!skip_deltastep)
            {
                ds_duration += std::chrono::duration_cast<std::chrono::milliseconds>(
                                   time_ds_end - time_dij_end)
                                   .count();
            }
            if (!skip_bellmanford)
            {
                bf_duration += std::chrono::duration_cast<std::chrono::milliseconds>(
                                   time_bf_end - time_ds_end)
                                   .count();
            }
            if (!skip_nearfar)
            {
                nf_duration += std::chrono::duration_cast<std::chrono::milliseconds>(
                                   time_nf_end - time_bf_end)
                                   .count();
            }
            if (!skip_dial)
            {
                dl_duration += std::chrono::duration_cast<std::chrono::milliseconds>(
                                   time_dl_end - time_nf_end)
                                   .count();
            }

            if (!skip_deltastep && dist != expected_dist)
            {
                common::log_error()
                    << "Error: DeltaStep distance " << src_nodes[i] << "->" << dst_nodes[i]
                    << " mismatch. expected: " << expected_dist << " actual: " << dist << '\n';
            }
            if (!skip_bellmanford && bf_dist != expected_dist)
            {
                common::log_error()
                    << "Error: BellmanFord distance " << src_nodes[i] << "->" << dst_nodes[i]
                    << " mismatch. expected: " << expected_dist << " actual: " << bf_dist << '\n';
            }
            if (!skip_nearfar && nf_dist != expected_dist)
            {
                common::log_error()
                    << "Error: NearFar distance " << src_nodes[i] << "->" << dst_nodes[i]
                    << " mismatch. expected: " << expected_dist << " actual: " << nf_dist << '\n';
            }
            if (!skip_dial && dl_dist != expected_dist)
            {
                common::log_error()
                    << "Error: Dial distance " << src_nodes[i] << "->" << dst_nodes[i]
                    << " mismatch. expected: " << expected_dist << " actual: " << dl_dist << '\n';
            }
            checksum += dist;
        }
        auto num_reachable = num_queries - num_unreachable;
        auto &log = common::log();
        log << "Processed " << num_reachable << " queries (" << num_unreachable << " unreachable)";
        if (!skip_dijkstra)
            log << " " << (dij_duration / num_reachable) << "ms/req (dijkstra)";
        if (!skip_dial)
            log << " " << (dl_duration / num_reachable) << "ms/req (dial"
                << (!skip_dijkstra ? " " + std::to_string(dij_duration / static_cast<double>(dl_duration)) : "")
                << ")";
        if (!skip_deltastep)
            log << " " << (ds_duration / num_reachable) << "ms/req (deltastep"
                << (!skip_dijkstra ? " " + std::to_string(dij_duration / static_cast<double>(ds_duration)) : "")
                << ")";
        if (!skip_bellmanford)
            log << " " << (bf_duration / num_reachable) << "ms/req (bellmanford"
                << (!skip_dijkstra ? " " + std::to_string(dij_duration / static_cast<double>(bf_duration)) : "")
                << ")";
        if (!skip_nearfar)
            log << " " << (nf_duration / num_reachable) << "ms/req (nearfar"
                << (!skip_dijkstra ? " " + std::to_string(dij_duration / static_cast<double>(nf_duration)) : "")
                << ")";
        log << '\n';
        common::log() << "Checksum: " << (checksum / num_reachable) << '\n';

#ifdef ENABLE_STATISTICS
        common::log() << "Statistics: " << std::endl
                      << common::Statistics::get().summary() << gpu_statistics.summary()
                      << std::endl;
#endif
    }

    return 0;
}
