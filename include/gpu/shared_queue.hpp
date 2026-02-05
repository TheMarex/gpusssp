#ifndef GPUSSSP_GPU_SHARED_QUEUE_HPP
#define GPUSSSP_GPU_SHARED_QUEUE_HPP

#include <mutex>
#include <vulkan/vulkan.hpp>

namespace gpusssp::gpu
{

class SharedQueue
{
  public:
    SharedQueue() : m_queue(nullptr) {}
    explicit SharedQueue(vk::Queue queue) : m_queue(queue) {}

    vk::Result submit(uint32_t submitCount, const vk::SubmitInfo *pSubmits, vk::Fence fence = nullptr)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_queue.submit(submitCount, pSubmits, fence);
    }

    void submit(const vk::ArrayProxy<const vk::SubmitInfo> &submits, vk::Fence fence = nullptr)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.submit(submits, fence);
    }

    vk::Result presentKHR(const vk::PresentInfoKHR &presentInfo)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_queue.presentKHR(presentInfo);
    }

    void waitIdle()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.waitIdle();
    }

    vk::Queue unwrap() const { return m_queue; }

  private:
    vk::Queue m_queue;
    mutable std::mutex m_mutex;
};

} // namespace gpusssp::gpu

#endif
