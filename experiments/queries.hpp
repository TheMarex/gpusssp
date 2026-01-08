#ifndef GPUSSSP_EXPERIMENTS_QUERIES_HPP
#define GPUSSSP_EXPERIMENTS_QUERIES_HPP

#include "common/serialization.hpp"

#include <string>
#include <vector>
#include <cstdint>

namespace gpusssp::experiments {
  struct Query {
    uint32_t from;
    uint32_t to;
  };

  inline auto read_queries(const std::string &base_path)
  {
      std::vector<Query> queries;
      common::BinaryReader reader(base_path + "/queries");
      common::serialization::read(reader, queries);
      return queries;
  }

  inline void write_queries(const std::string &base_path,
                                const std::vector<Query> &queries)
  {
      common::BinaryWriter writer(base_path + "/queries");
      common::serialization::write(writer, queries);
  }
}

#endif
