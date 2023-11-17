#include "network.hpp"

namespace nnet {
  float* NeuralNetwork::forward(float* input) {
    float* output = input;

    for (auto i = this->layers.begin(); i != this->layers.end(); i++) {
      float* layerOutput = i->forward(output);
      delete[] output;
      output = layerOutput;
    }

    return output;
  }

  void NeuralNetwork::addLayer(Layer* layer) {
    this->layers.push_back(*layer);
  }
}
