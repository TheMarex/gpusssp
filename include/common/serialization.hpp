#ifndef GPUSSSP_COMMON_SERILIAZATION_HPP
#define GPUSSSP_COMMON_SERILIAZATION_HPP

#include "common/binary.hpp"

#include <vector>

namespace gpusssp {
namespace common {
namespace serialization {
    template<typename T>
    void read(BinaryReader &reader, std::vector<T> &vector)
    {
        std::uint64_t count;
        reader.read(count, 1);
        vector.resize(count);
        reader.read(*vector.data(), count);
    }

    template<typename T>
    void write(BinaryWriter &writer, const std::vector<T> &vector)
    {
        writer.write(vector.size(), 1);
        writer.write(*vector.data(), vector.size());
    }
}
}
}

#endif
