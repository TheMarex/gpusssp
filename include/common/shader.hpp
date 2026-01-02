#ifndef GPUSSSP_COMMON_SHADER_HPP
#define GPUSSSP_COMMON_SHADER_HPP

#include <vector>
#include <fstream>
#include <cassert>
#include <cstdint>
#include <cstring>

namespace gpusssp::common {

// Utility: read SPIR-V file
inline std::vector<uint32_t> read_spv(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    assert(file.is_open());
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    size_t wordCount = bytes.size() / 4;
    std::vector<uint32_t> spv(wordCount);
    std::memcpy(spv.data(), bytes.data(), wordCount * 4);
    return spv;
}

}

#endif
