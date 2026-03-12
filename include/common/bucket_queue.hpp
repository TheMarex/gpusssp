#ifndef GPUSSSP_COMMON_BUCKET_QUEUE_HPP
#define GPUSSSP_COMMON_BUCKET_QUEUE_HPP

#include "common/circular_vector.hpp"
#include "common/constants.hpp"
#include "common/lazy_clear_vector.hpp"
#include "common/queue.hpp"
#include "common/statistics.hpp"

#include <cassert>
#include <vector>

namespace gpusssp::common
{

class BucketQueue
{
    struct BucketEntry
    {
        IDKeyPair p;
        unsigned prev;
        unsigned next;
    };

  public:
    explicit BucketQueue(unsigned id_count, unsigned initial_bound)
        : buckets(initial_bound, INVALID_ID), id_entry(id_count, INVALID_ID)
    {
    }

    //! Returns whether the queue is empty. Equivalent to checking whether size() returns 0.
    [[nodiscard]] bool empty() const { return size() == 0; }

    //! Returns the number of elements in the queue.
    [[nodiscard]] unsigned size() const { return entries.size() - free_entries.size(); }

    //! Checks whether an element is in the queue.
    bool contains_id(unsigned id)
    {
        assert(id < id_entry.size());
        return id_entry[id] != INVALID_ID;
    }

    //! Removes all elements from the queue.
    void clear()
    {
        id_entry.clear();
        buckets.clear();
        entries.clear();
        free_entries.clear();
    }

    //! Returns the smallest element key pair without removing it from the queue.
    [[nodiscard]] IDKeyPair peek() const
    {
        auto min_entry_id = buckets.front();
        return entries[min_entry_id].p;
    }

    //! Returns the smallest element key pair and removes it from the queue.
    IDKeyPair pop()
    {
        Statistics::get().count(StatisticsEvent::QUEUE_POP);
        auto &min_entry_index = buckets.front();
        auto &min_entry = entries[min_entry_index];
        auto new_min_entry_index = min_entry.next;
        IDKeyPair p = min_entry.p;

        id_entry[p.id] = INVALID_ID;
        free_entries.push_back(min_entry_index);

        if (new_min_entry_index == INVALID_ID)
        {
            buckets.pop_front();
        }
        else
        {
            min_entry_index = new_min_entry_index;
            auto &new_min_entry = entries[new_min_entry_index];
            new_min_entry.prev = INVALID_ID;
        }

        return p;
    }

    //! Inserts a element key pair.
    //! Undefined if the element is part of the queue.
    void push(IDKeyPair p)
    {
        Statistics::get().count(StatisticsEvent::QUEUE_PUSH);

        const unsigned next_entry_index = buckets.peek(p.key);

        unsigned entry_index = entries.size();
        if (!free_entries.empty())
        {
            entry_index = free_entries.back();
            free_entries.pop_back();
            entries[entry_index] =
                BucketEntry{.p = p, .prev = INVALID_ID, .next = next_entry_index};
        }
        else
        {
            entries.emplace_back(p, INVALID_ID, next_entry_index);
        }
        id_entry[p.id] = entry_index;

        buckets.update(p.key, entry_index);
        if (next_entry_index != INVALID_ID)
        {
            entries[next_entry_index].prev = entry_index;
        }
    }

    //! Updates the key of an element if the new key is smaller than the old key.
    //! Does nothing if the new key is larger.
    //! Undefined if the element is not part of the queue.
    bool decrease_key(IDKeyPair p)
    {
        Statistics::get().count(StatisticsEvent::QUEUE_DECREASE_KEY);

        auto entry_index = id_entry[p.id];
        assert(entry_index != INVALID_ID);
        auto &entry = entries[entry_index];
        auto old_key = entry.p.key;
        if (p.key < old_key)
        {
            unlink_entry(old_key, entry, entry_index);
            insert_entry(p.key, entry, entry_index);

            entry.p.key = p.key;

            return true;
        }

        return false;
    }

  private:
    void unlink_entry(unsigned key, BucketEntry &entry, unsigned entry_index)
    {
        const unsigned head_entry_index = buckets.peek(key);
        if (entry.prev != INVALID_ID)
        {
            entries[entry.prev].next = entry.next;
        }
        if (entry.next != INVALID_ID)
        {
            entries[entry.next].prev = entry.prev;
        }
        if (head_entry_index == entry_index)
        {
            buckets.update(key, entry.next);
        }
    }

    void insert_entry(unsigned key, BucketEntry &entry, unsigned entry_index)
    {
        auto head_entry_index = buckets.peek(key);
        auto next_entry_index = head_entry_index;

        entry.prev = INVALID_ID;
        entry.next = next_entry_index;
        if (next_entry_index != INVALID_ID)
        {
            entries[next_entry_index].prev = entry_index;
        }
        buckets.update(static_cast<std::size_t>(key), entry_index);
    }

    CircularVector<unsigned> buckets;
    LazyClearVector<unsigned> id_entry;
    std::vector<BucketEntry> entries;
    std::vector<unsigned> free_entries;
};

} // namespace gpusssp::common

#endif
