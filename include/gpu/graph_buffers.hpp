#ifndef GPUSSSP_GPU_GRAPH_BUFFERS_HPP
#define GPUSSSP_GPU_GRAPH_BUFFERS_HPP

namespace gpusssp::gpu {

template<typename GraphT>
class GraphBuffers {
public:
  GraphBuffers(const GraphT& graph) : graph(graph) {
  }

private:
  const GraphT& graph;
};

}

#endif
