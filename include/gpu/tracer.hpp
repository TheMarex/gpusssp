#ifndef GPUSSSP_GPU_TRACER_HPP
#define GPUSSSP_GPU_TRACER_HPP

#ifdef ENABLE_TRACING

#include "common/logger.hpp"
#include <chrono>
#include <condition_variable>
#include <mutex>

namespace gpusssp::gpu
{

class Tracer
{
  public:
    Tracer() : should_continue(false), ready_to_step(false), finished(false) {}

    void start()
    {
        common::log() << "Starting run" << std::endl;
        {
            std::unique_lock<std::mutex> lock(step_mtx);
            should_continue = false;
            ready_to_step = false;
        }
        {
            std::unique_lock<std::mutex> lock(run_mtx);
            finished = false;
        }
    }

    void signal_and_wait()
    {
        if (should_continue)
        {
            return;
        }

        {
            common::log() << "Signaling ready to step..." << std::endl;
            std::unique_lock<std::mutex> lock(render_mtx);
            ready_to_step = false;
            cv_render.notify_one();
        }

        {
            common::log() << "Waiting to step..." << std::endl;
            std::unique_lock<std::mutex> lock(step_mtx);
            cv_step.wait(lock, [this] { return ready_to_step || should_continue; });
        }
    }

    void step()
    {
        {
            std::unique_lock<std::mutex> lock(run_mtx);
            if (finished)
            {
                return;
            }
        }

        std::unique_lock<std::mutex> lock(step_mtx);
        ready_to_step = true;
        cv_step.notify_one();

        common::log() << "Stepped!" << std::endl;
    }

    void continue_to_end()
    {
        {
            std::unique_lock<std::mutex> lock(run_mtx);
            if (finished)
            {
                return;
            }
        }

        std::unique_lock<std::mutex> lock(step_mtx);
        should_continue = true;
        cv_step.notify_one();

        common::log() << "Continuing." << std::endl;
    }

    bool wait_for_signal(int timeout_ms = 100)
    {
        std::unique_lock<std::mutex> lock(render_mtx);
        return cv_render.wait_for(lock, std::chrono::milliseconds(timeout_ms)) ==
               std::cv_status::no_timeout;
    }

    bool is_finished() const { return finished; }

    void finish()
    {
        {
            std::unique_lock<std::mutex> lock(run_mtx);
            finished = true;
            cv_run.notify_all();
        }
        common::log() << "Finished." << std::endl;
    }

    void wait_for_finished()
    {
        common::log() << "Waiting to finish..." << std::endl;
        std::unique_lock<std::mutex> lock(run_mtx);
        cv_run.wait(lock, [this] { return this->finished; });
    }

  private:
    std::mutex step_mtx;
    std::mutex render_mtx;
    std::mutex run_mtx;
    std::condition_variable cv_run;
    std::condition_variable cv_step;
    std::condition_variable cv_render;
    bool should_continue;
    bool finished;
    bool ready_to_step;
};

} // namespace gpusssp::gpu

#else

namespace gpusssp::gpu
{

struct Tracer
{
    void signal_and_wait() {}
    void step() {}
    void continue_to_end() {}
    bool wait_for_signal(int = 100) { return false; }
    void wait_for_finished() {}
    bool is_finished() const { return false; }
    void finish() {}
    void start() {}
};

} // namespace gpusssp::gpu

#endif

#endif
