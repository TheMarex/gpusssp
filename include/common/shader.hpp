#ifndef GPUSSSP_COMMON_SHADER_HPP
#define GPUSSSP_COMMON_SHADER_HPP

#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

namespace gpusssp::common
{

// Utility: read SPIR-V file
inline std::vector<uint32_t> read_spv(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    assert(file.is_open());
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    size_t word_count = bytes.size() / 4;
    std::vector<uint32_t> spv(word_count);
    std::memcpy(spv.data(), bytes.data(), word_count * 4);
    return spv;
}

} // namespace gpusssp::common

#endif
