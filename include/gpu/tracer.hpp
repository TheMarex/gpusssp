#ifndef GPUSSSP_GPU_TRACER_HPP
#define GPUSSSP_GPU_TRACER_HPP

#ifdef ENABLE_TRACING

#include "common/logger.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>

namespace gpusssp::gpu
{

template <typename PayloadT> class Tracer
{
  public:
    Tracer()
        : should_continue(false), ready_to_step(false), finished(true), auto_play_enabled(false),
          auto_play_interval_ms(500)
    {
    }

    void start()
    {
        common::log() << "Starting run" << std::endl;
        {
            std::unique_lock<std::mutex> lock(step_mtx);
            should_continue = false;
            ready_to_step = false;
            maybe_payload.emplace();
        }
        {
            std::unique_lock<std::mutex> lock(run_mtx);
            finished = false;
        }
    }

    void signal_and_wait(PayloadT payload)
    {
        if (should_continue)
        {
            return;
        }

        {
            common::log() << "Signaling ready to step..." << std::endl;
            std::unique_lock<std::mutex> lock(render_mtx);
            maybe_payload.emplace(std::move(payload));
            ready_to_step = false;
            cv_render.notify_one();
        }

        {
            common::log() << "Waiting to step..." << std::endl;
            std::unique_lock<std::mutex> lock(step_mtx);
            if (auto_play_enabled)
            {
                cv_step.wait_for(lock,
                                 std::chrono::milliseconds(auto_play_interval_ms),
                                 [this] { return ready_to_step || should_continue; });
            }
            else
            {
                cv_step.wait(lock, [this] { return ready_to_step || should_continue; });
            }
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

        {
            std::unique_lock<std::mutex> lock(step_mtx);
            should_continue = true;
            cv_step.notify_one();
        }

        common::log() << "Continuing..." << std::endl;

        {
            std::unique_lock<std::mutex> lock(run_mtx);
            cv_run.wait(lock, [this] { return this->finished; });
        }
    }

    bool wait_for_signal(int timeout_ms = 100)
    {
        std::unique_lock<std::mutex> lock(render_mtx);
        return cv_render.wait_for(lock, std::chrono::milliseconds(timeout_ms)) ==
               std::cv_status::no_timeout;
    }

    std::optional<PayloadT> payload()
    {
        std::unique_lock<std::mutex> lock(render_mtx);
        return maybe_payload;
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
        std::unique_lock<std::mutex> lock(run_mtx);
        cv_run.wait(lock, [this] { return this->finished; });
    }

    void start_auto_play(uint32_t interval_ms = 500)
    {
        auto_play_interval_ms = interval_ms;
        auto_play_enabled = true;
        common::log() << "Auto-play started (interval: " << interval_ms << "ms)" << std::endl;
    }

    void stop_auto_play()
    {
        auto_play_enabled = false;
        common::log() << "Auto-play stopped" << std::endl;
    }

    void set_auto_play_speed(uint32_t interval_ms)
    {
        auto_play_interval_ms = interval_ms;
        common::log() << "Auto-play speed changed to " << interval_ms << "ms" << std::endl;
    }

    bool is_auto_playing() const { return auto_play_enabled; }

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
    std::atomic<bool> auto_play_enabled;
    std::atomic<uint32_t> auto_play_interval_ms;
    std::optional<PayloadT> maybe_payload;
};

} // namespace gpusssp::gpu

#else

namespace gpusssp::gpu
{

template <typename PayloadT> struct Tracer
{
    void signal_and_wait(PayloadT) {}
    void step() {}
    void continue_to_end() {}
    bool wait_for_signal(int = 100) { return false; }
    void wait_for_finished() {}
    bool is_finished() const { return false; }
    void finish() {}
    void start() {}
    std::optional<PayloadT> payload() { return {}; }
    void start_auto_play(uint32_t = 500) {}
    void stop_auto_play() {}
    void set_auto_play_speed(uint32_t) {}
    bool is_auto_playing() const { return false; }
};

} // namespace gpusssp::gpu

#endif

#endif
