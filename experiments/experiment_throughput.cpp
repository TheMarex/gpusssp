#include "experiment_util.hpp"
#include "queries.hpp"
#include "common/csv.hpp"
#include "common/dial.hpp"
#include "common/dijkstra.hpp"
#include "common/files.hpp"
#include "common/logger.hpp"
#include "common/weighted_graph.hpp"
#include "gpu/graph_buffers.hpp"
#include "gpu/nearfar.hpp"
#include "gpu/nearfar_buffers.hpp"
#include "gpu/shared_queue.hpp"
#include "gpu/vulkan_context.hpp"

#include <argparse/argparse.hpp>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>
#include <random>
#include <thread>
#include <vector>
#include <vulkan/vulkan.hpp>

using namespace gpusssp;

namespace
{
class AlgorithmWorker
{
  public:
    virtual ~AlgorithmWorker() = default;

    void submit(const std::vector<experiments::Query> &queries)
    {
        {
            std::lock_guard lock(mutex);
            current_queries = &queries;
            is_done = false;
            has_work = true;
        }
        cv.notify_one();
    }

    void wait()
    {
        std::unique_lock lock(mutex);
        cv.wait(lock, [this] { return is_done; });
    }

    void stop_and_join()
    {
        thread.request_stop();
        cv.notify_all();
        if (thread.joinable())
        {
            thread.join();
        }
    }

  protected:
    AlgorithmWorker()
    {
        thread = std::jthread(
            [this](std::stop_token st)
            {
                while (!st.stop_requested())
                {
                    std::unique_lock lock(mutex);
                    cv.wait(lock, [this, &st] { return has_work || st.stop_requested(); });
                    if (st.stop_requested())
                    {
                        break;
                    }

                    run();

                    has_work = false;
                    is_done = true;
                    lock.unlock();
                    cv.notify_all();
                }
            });
    }

    virtual void run() = 0;

    std::mutex mutex;
    std::condition_variable cv;
    bool has_work = false;
    bool is_done = true;
    const std::vector<experiments::Query> *current_queries = nullptr;
    std::jthread thread;
};

template <typename GraphT> class NearFarWorker : public AlgorithmWorker
{
  public:
    NearFarWorker(const gpu::GraphBuffers<GraphT> &graph_buffers,
                  gpu::SharedQueue &queue,
                  vk::Device device,
                  const vk::PhysicalDeviceMemoryProperties &mem_props,
                  uint32_t delta,
                  uint32_t batch_size)
        : queue(queue), device(device),
          cmd_pool(
              device.createCommandPool({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, 0})),
          nearfar_buffers(graph_buffers.num_nodes(), device, mem_props),
          statistics(device, mem_props),
          nearfar(graph_buffers, nearfar_buffers, device, statistics, delta, batch_size)
    {
        nearfar.initialize(cmd_pool);
    }

    ~NearFarWorker() override
    {
        this->stop_and_join();
        device.destroyCommandPool(cmd_pool);
    }

  private:
    void run() override
    {
        for (const auto &query : *this->current_queries)
        {
            nearfar.run(cmd_pool, queue, query.from, query.to);
        }
    }

    gpu::SharedQueue &queue;
    vk::Device device;
    vk::CommandPool cmd_pool;
    gpu::NearFarBuffers nearfar_buffers;
    gpu::Statistics statistics;
    gpu::NearFar<GraphT> nearfar;
};

template <typename GraphT> class DijkstraWorker : public AlgorithmWorker
{
  public:
    explicit DijkstraWorker(const GraphT &graph)
        : graph(graph), min_queue(graph.num_nodes()), costs(graph.num_nodes(), common::INF_WEIGHT),
          settled(graph.num_nodes(), false)
    {
    }

    ~DijkstraWorker() override { this->stop_and_join(); }

  private:
    void run() override
    {
        for (const auto &query : *this->current_queries)
        {
            common::dijkstra(query.from, query.to, graph, min_queue, costs, settled);
        }
    }

    const GraphT &graph;
    common::MinIDQueue min_queue;
    common::CostVector<GraphT> costs;
    std::vector<bool> settled;
};

template <typename GraphT> class DialWorker : public AlgorithmWorker
{
  public:
    explicit DialWorker(const GraphT &graph)
        : graph(graph), bucket_queue(graph.num_nodes(), 32 * 1024),
          costs(graph.num_nodes(), common::INF_WEIGHT), settled(graph.num_nodes(), false)
    {
    }

    ~DialWorker() override { this->stop_and_join(); }

  private:
    void run() override
    {
        for (const auto &query : *this->current_queries)
        {
            common::dial(query.from, query.to, graph, bucket_queue, costs, settled);
        }
    }

    const GraphT &graph;
    common::BucketQueue bucket_queue;
    common::CostVector<GraphT> costs;
    std::vector<bool> settled;
};

template <typename GraphT>
std::vector<std::unique_ptr<AlgorithmWorker>>
create_workers(const std::string &algorithm,
               uint32_t max_threads,
               const GraphT &graph,
               const gpu::GraphBuffers<GraphT> &graph_buffers,
               gpu::SharedQueue &shared_queue,
               vk::Device device,
               const vk::PhysicalDeviceMemoryProperties &mem_props,
               uint32_t delta,
               uint32_t batch_size)
{
    std::vector<std::unique_ptr<AlgorithmWorker>> workers;
    if (algorithm == "nearfar")
    {
        for (uint32_t i = 0; i < max_threads; ++i)
        {
            workers.push_back(std::make_unique<NearFarWorker<GraphT>>(
                graph_buffers, shared_queue, device, mem_props, delta, batch_size));
        }
    }
    else if (algorithm == "dijkstra")
    {
        for (uint32_t i = 0; i < max_threads; ++i)
        {
            workers.push_back(std::make_unique<DijkstraWorker<GraphT>>(graph));
        }
    }
    else if (algorithm == "dial")
    {
        for (uint32_t i = 0; i < max_threads; ++i)
        {
            workers.push_back(std::make_unique<DialWorker<GraphT>>(graph));
        }
    }
    else
    {
        throw std::runtime_error("Unknown algorithm: " + algorithm);
    }

    return workers;
}

} // namespace

int main(int argc, char **argv)
{
    argparse::ArgumentParser program("experiment_throughput", "1.0.0");
    program.add_description("Run algorithm throughput benchmark.");

    program.add_argument("graph_path").help("path to preprocessed graph data (without extension)");
    program.add_argument("xp_path").help("base path for experiment output");

    program.add_argument("-n", "--name")
        .default_value(std::string("throughput"))
        .help("experiment name");

    program.add_argument("-a", "--algorithm")
        .default_value(std::string("nearfar"))
        .help("algorithm to benchmark: nearfar, dijkstra, dial");

    program.add_argument("--delta").default_value(3600u).scan<'u', uint32_t>().help(
        "delta parameter for algorithms that support it");

    program.add_argument("--batch-size")
        .default_value(64u)
        .scan<'u', uint32_t>()
        .help("batch size for GPU processing");

    program.add_argument("--max-threads")
        .default_value(8u)
        .scan<'u', uint32_t>()
        .help("maximum number of worker threads");

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
    std::string algorithm = program.get("--algorithm");
    auto delta = program.get<uint32_t>("--delta");
    auto batch_size = program.get<uint32_t>("--batch-size");
    auto max_threads = program.get<uint32_t>("--max-threads");

    common::log() << "Loading graph from: " << graph_base_path << '\n';
    auto graph = common::files::read_weighted_graph<uint32_t>(graph_base_path);
    common::log() << "Graph loaded: " << graph.num_nodes() << " nodes." << '\n';

    common::log() << "Loading queries from: " << graph_base_path << "/queries.csv" << '\n';
    auto all_queries = experiments::read_queries(graph_base_path);
    common::log() << "Loaded " << all_queries.size() << " queries." << '\n';

    // Group queries by rank
    std::map<uint8_t, std::vector<experiments::Query>> queries_by_rank;
    for (const auto &query : all_queries)
    {
        if (query.rank)
        {
            queries_by_rank[*query.rank].push_back(query);
        }
    }

    gpu::VulkanContext vk_ctx("ThroughputExperiment", gpu::detail::select_device());
    uint64_t timestamp = experiments::get_unix_timestamp();
    std::string query_hash = experiments::hash_queries_content(all_queries);
    std::string device_id = experiments::hash_device_name(vk_ctx.device_name());
    std::string graph_name = experiments::extract_graph_name(graph_base_path);

    experiments::create_experiment_directories(
        xp_base_path, xp_name, graph_name, query_hash, device_id);

    std::ostringstream variant_stream;
    variant_stream << algorithm << "_delta" << delta << "_batch" << batch_size << "_maxthreads"
                   << max_threads;
    std::string variant = variant_stream.str();

    std::string output_filename = experiments::generate_experiment_filename(
        xp_base_path, xp_name, graph_name, query_hash, device_id, timestamp, variant);
    common::log() << "Output file: " << output_filename << '\n';

    auto device = vk_ctx.device();
    auto queue = vk_ctx.queue();
    auto &shared_queue = vk_ctx.shared_queue();
    auto cmd_pool = vk_ctx.command_pool();

    common::CSVWriter<uint8_t, uint32_t, double, uint32_t, uint64_t> writer(output_filename);
    writer.write_header({"rank", "threads", "throughput", "total_queries", "total_time_us"});

    gpu::GraphBuffers graph_buffers(graph, device, vk_ctx.memory_properties(), cmd_pool, queue);

    auto workers = create_workers(algorithm,
                                  max_threads,
                                  graph,
                                  graph_buffers,
                                  shared_queue,
                                  device,
                                  vk_ctx.memory_properties(),
                                  delta,
                                  batch_size);

    for (const auto &[rank, queries] : queries_by_rank)
    {
        common::log() << "Processing rank " << static_cast<uint32_t>(rank) << " with "
                      << queries.size() << " queries..." << '\n';

        std::vector<std::vector<experiments::Query>> thread_queries(max_threads, queries);
        std::random_device rd;
        for (auto &t_queries : thread_queries)
        {
            std::mt19937 g(rd());
            std::shuffle(t_queries.begin(), t_queries.end(), g);
        }

        for (uint32_t num_threads = 1; num_threads <= max_threads; num_threads *= 2)
        {
            common::log() << "  Running with " << num_threads << " threads...";

            auto start_time = std::chrono::high_resolution_clock::now();

            for (uint32_t t = 0; t < num_threads; ++t)
            {
                workers[t]->submit(thread_queries[t]);
            }

            for (uint32_t t = 0; t < num_threads; ++t)
            {
                workers[t]->wait();
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration_us =
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)
                    .count();

            auto total_queries = static_cast<uint32_t>(queries.size() * num_threads);
            double throughput = static_cast<double>(total_queries) / duration_us * 1000000.0;
            common::log() << " throughput " << throughput << " qps" << '\n';

            writer.write({rank, num_threads, throughput, total_queries, duration_us});
        }
    }

    for (auto &worker : workers)
    {
        worker->stop_and_join();
    }

    common::log() << "Done." << '\n';

    return 0;
}
