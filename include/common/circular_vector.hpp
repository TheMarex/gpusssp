#ifndef GPUSSSP_COMMON_CIRCULAR_VECTOR_HPP
#define GPUSSSP_COMMON_CIRCULAR_VECTOR_HPP

#include "common/lazy_clear_vector.hpp"
#include <bit>
#include <cassert>
#include <limits>

namespace gpusssp::common
{

// This class implements an vector that can store elements v[i..i+bounds] for any i.
// If you try to access and element at index j and j is outside this range,
// it will be undefined behavior.
// You can check if a value is in range with in_bounds() though.
template <typename T> class CircularVector
{
  public:
    explicit CircularVector(unsigned bound, const T &default_value) // NOLINT
        : default_value{default_value}, data(std::bit_ceil(bound), default_value),
          index_mask{data.size() - 1}
    {
    }

    void clear()
    {
        data.clear();
        min_index = std::numeric_limits<std::size_t>::max();
        max_index = 0;
    }

    [[nodiscard]] T peek(const std::size_t index) const
    {
        assert(in_bounds(index));
        return data[local_index_unchecked(index)];
    }

    //! Updates a value at index, calls clear(index) and mark(index) on-demand
    void update(const std::size_t index, const T &value)
    {
        assert(in_bounds(index));
        data[local_index_unchecked(index)] = value;

        // If we "unset" a value we need to update the min/max
        if (value == default_value)
        {
            if (index == min_index)
            {
                min_index++;
                update_min_index();
            }
            else if (index == max_index)
            {
                max_index--;
                update_max_index();
            }
        }
        else if (!empty())
        {
            min_index = std::min(min_index, index);
            max_index = std::max(max_index, index);
        }
        else
        {
            min_index = index;
            max_index = index;
        }
    }

    [[nodiscard]] std::size_t size() const
    {
        if (empty())
            return 0;
        return max_index - min_index + 1;
    }

    [[nodiscard]] bool empty() const { return min_index > max_index; }

    [[nodiscard]] std::size_t front_index() const { return min_index; }
    [[nodiscard]] std::size_t back_index() const { return max_index; }

    //! Returns first element, undefined behavior if empty.
    [[nodiscard]] const T &front() const
    {
        assert(!empty());
        return data[local_index_unchecked(min_index)];
    }

    [[nodiscard]] T &front()
    {
        assert(!empty());
        return data[local_index_unchecked(min_index)];
    }

    //! Returns last element, undefined behavior if empty.
    [[nodiscard]] const T &back() const
    {
        assert(!empty());
        return data[local_index_unchecked(max_index)];
    }

    [[nodiscard]] T &back()
    {
        assert(!empty());
        return data[local_index_unchecked(max_index)];
    }

    [[nodiscard]] bool in_bounds(const std::size_t index) const
    {
        // For empty vectors any index is in bounds because we can move it
        if (empty())
        {
            return true;
        }

        //   <--------bounds------->
        //             <--------bounds------->
        //             |-----------|
        //           min_index   max_index
        if (index < min_index)
        {
            return index + data.size() > max_index;
        }
        else if (index > max_index)
        {
            return index < min_index + data.size();
        }
        else
        {
            return true;
        }
    }

    [[nodiscard]] bool empty(const std::size_t index) const
    {
        assert(in_bounds(index));
        return data[local_index_unchecked(index)] == default_value;
    }

    void pop_front()
    {
        assert(!empty());
        data[local_index_unchecked(min_index++)] = default_value;

        update_min_index();
    }

    void push_back(const T &value) { data[local_index_unchecked(++max_index)] = value; }

  private:
    // Find the next non-empty value
    void update_min_index()
    {
        while (min_index <= max_index && empty(min_index))
        {
            min_index++;
        }
    }

    void update_max_index()
    {
        while (min_index <= max_index && empty(max_index))
        {
            max_index--;
        }
    }

    // This function does not implement bounds checks
    [[nodiscard]] std::size_t local_index_unchecked(std::size_t index) const
    {
        // Since the distance between min and max is less than data.size()
        // we can guarantee that this never aliases.
        auto local_index = index & index_mask;
        return local_index;
    }

    // The maximum difference between min_value and max_value can be bound
    std::size_t min_index = std::numeric_limits<std::size_t>::max();
    std::size_t max_index = 0;

    const T default_value;
    LazyClearVector<T> data;
    std::size_t index_mask;
};

} // namespace gpusssp::common

#endif
