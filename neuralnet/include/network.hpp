#pragma once

#include <vector>
#include "layer.hpp"

namespace nnet {
  class NeuralNetwork {
  private:
    std::vector<Layer> layers;
  public:
    float* forward(float* input);

    void addLayer(Layer *layer);
  };
}
