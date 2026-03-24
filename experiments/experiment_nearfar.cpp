#include "experiment_util.hpp"
#include "queries.hpp"
#include "common/csv.hpp"
#include "common/files.hpp"
#include "common/logger.hpp"
#include "common/progress_bar.hpp"
#include "common/statistics.hpp"
#include "common/weighted_graph.hpp"
#include "gpu/statistics.hpp"

#include "gpu/graph_buffers.hpp"
#include "gpu/nearfar.hpp"
#include "gpu/nearfar_buffers.hpp"
#include "gpu/shared_queue.hpp"
#include "gpu/vulkan_context.hpp"

#include <argparse/argparse.hpp>
#include <chrono>
#include <cstdint>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <vulkan/vulkan.hpp>

using namespace gpusssp;

namespace
{
vk::CommandPool create_command_pool(vk::Device device, uint32_t queue_family_index)
{
    return device.createCommandPool(
        {vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queue_family_index});
}

using ResultWriterT =
    common::CSVWriter<uint32_t, uint32_t, std::optional<uint8_t>, uint32_t, std::vector<uint64_t>>;

template <typename QueueT>
void run_queries(const std::vector<experiments::Query> &queries,
                 const gpu::GraphBuffers<common::WeightedGraph<uint32_t>> &graph_buffers,
                 QueueT &queue,
                 vk::Device device,
                 const vk::PhysicalDeviceMemoryProperties &mem_props,
                 uint32_t delta,
                 uint32_t batch_size,
                 ResultWriterT &writer,
                 const std::vector<std::string> &metrics,
                 common::ProgressBar &progress_bar)
{
    auto cmd_pool = create_command_pool(device, 0u);
    gpu::NearFarBuffers nearfar_buffers(graph_buffers.num_nodes(), device, mem_props);
    gpu::Statistics statistics(device, mem_props);
    gpu::NearFar nearfar(graph_buffers, nearfar_buffers, device, statistics, delta, batch_size);
    nearfar.initialize(cmd_pool);

    std::vector<uint64_t> metric_values;
    metric_values.reserve(metrics.size());

    for (const auto &query : queries)
    {
        metric_values.clear();

        auto start_time = std::chrono::high_resolution_clock::now();
        auto start_edges = statistics.value(gpu::StatisticsEvent::NEARFAR_EDGES_RELAXED);

        auto dist = nearfar.run(cmd_pool, queue, query.from, query.to);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        auto edges_relaxed =
            statistics.value(gpu::StatisticsEvent::NEARFAR_EDGES_RELAXED) - start_edges;

        for (const auto &metric : metrics)
        {
            if (metric == "time")
            {
                metric_values.push_back(duration);
            }
            else if (metric == "edges_relaxed")
            {
                metric_values.push_back(edges_relaxed);
            }
        }

        writer.write({query.from, query.to, query.rank, dist, metric_values});
        progress_bar.increment();
    }

    device.destroyCommandPool(cmd_pool);
}
} // namespace

int main(int argc, char **argv)
{
    argparse::ArgumentParser program("experiment_nearfar", "1.0.0");
    program.add_description("Run NearFar algorithm benchmark.");

    program.add_argument("graph_path").help("path to preprocessed graph data (without extension)");

    program.add_argument("xp_path").help("base path for experiment output");

    program.add_argument("-n", "--name")
        .default_value(std::string("compare_algorithm"))
        .help("experiment name");

    program.add_argument("--delta").default_value(3600u).scan<'u', uint32_t>().help(
        "delta parameter for near-far algorithm");

    program.add_argument("--batch-size")
        .default_value(64u)
        .scan<'u', uint32_t>()
        .help("batch size for GPU processing");

    program.add_argument("--metrics")
        .default_value(std::string("time"))
        .help("comma-separated metrics to capture: time, edges_relaxed");

    program.add_argument("--threads")
        .default_value(1u)
        .scan<'u', uint32_t>()
        .help("number of worker threads");

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception &err)
    {
        common::log_error() << err.what() << '\n' << program;
        return 1;
    }

    std::string graph_base_path = program.get("graph_path");
    std::string xp_base_path = program.get("xp_path");
    std::string xp_name = program.get("--name");
    auto delta = program.get<uint32_t>("--delta");
    auto batch_size = program.get<uint32_t>("--batch-size");
    auto num_threads = program.get<uint32_t>("--threads");

    auto metrics = experiments::parse_metrics(program.get("--metrics"));
    experiments::validate_metrics(metrics);

    common::log() << "Loading graph from: " << graph_base_path << '\n';
    auto graph = common::files::read_weighted_graph<uint32_t>(graph_base_path);
    common::log() << "Graph loaded: " << graph.num_nodes() << " nodes." << '\n';

    common::log() << "Loading queries from: " << graph_base_path << "/queries.csv" << '\n';
    auto queries = experiments::read_queries(graph_base_path);
    common::log() << "Loaded " << queries.size() << " queries." << '\n';

    gpu::VulkanContext vk_ctx("NearFarExperiment", gpu::detail::select_device());
    uint64_t timestamp = experiments::get_unix_timestamp();
    std::string query_hash = experiments::hash_queries_content(queries);
    std::string device_id = experiments::hash_device_name(vk_ctx.device_name());
    std::string graph_name = experiments::extract_graph_name(graph_base_path);

    std::ostringstream variant_stream;
    variant_stream << "nearfar_delta" << delta << "_batch" << batch_size << "_threads"
                   << num_threads;
    std::string variant = variant_stream.str();

    experiments::create_experiment_directories(
        xp_base_path, xp_name, graph_name, query_hash, device_id);
    std::string output_filename = experiments::generate_experiment_filename(
        xp_base_path, xp_name, graph_name, query_hash, device_id, timestamp, variant);
    common::log() << "Output file: " << output_filename << '\n';

    auto device = vk_ctx.device();
    auto queue = vk_ctx.queue();
    auto &shared_queue = vk_ctx.shared_queue();
    auto cmd_pool = vk_ctx.command_pool();

    std::vector<std::string> headers = {"from_node_id", "to_node_id", "rank", "distance"};
    for (const auto &metric : metrics)
    {
        headers.push_back(metric);
    }

    common::CSVWriter<uint32_t, uint32_t, std::optional<uint8_t>, uint32_t, std::vector<uint64_t>>
        writer(output_filename);
    writer.write_header(headers);

    common::log() << "Running queries with delta = " << delta << " batch_size = " << batch_size
                  << " num_threads = " << num_threads << "..." << '\n';

    {
        gpu::GraphBuffers graph_buffers(graph, device, vk_ctx.memory_properties(), cmd_pool, queue);

        common::ProgressBar progress_bar(queries.size());

        if (num_threads == 1)
        {
            run_queries(queries,
                        graph_buffers,
                        queue,
                        device,
                        vk_ctx.memory_properties(),
                        delta,
                        batch_size,
                        writer,
                        metrics,
                        progress_bar);
        }
        else
        {
            std::vector<std::vector<experiments::Query>> thread_queries(num_threads);
            for (size_t i = 0; i < queries.size(); ++i)
            {
                thread_queries[i % num_threads].push_back(queries[i]);
            }

            std::vector<std::jthread> threads;
            threads.reserve(num_threads);
            for (size_t t = 0; t < num_threads; ++t)
            {
                threads.emplace_back(run_queries<gpu::SharedQueue>,
                                     std::ref(thread_queries[t]),
                                     std::ref(graph_buffers),
                                     std::ref(shared_queue),
                                     device,
                                     vk_ctx.memory_properties(),
                                     delta,
                                     batch_size,
                                     std::ref(writer),
                                     std::ref(metrics),
                                     std::ref(progress_bar));
            }
        }

        common::log() << "Done." << '\n';
    }

    return 0;
}
