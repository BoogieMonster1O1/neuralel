#pragma once

#include <vector>
#include <stdexcept>
#include <cmath>
#include "layer.hpp"

namespace nnet {
  class NeuralNetwork {
  protected:
    std::vector<Layer*> layers;
  public:
    float* forward(float* input);

    virtual void addLayer(Layer *layer);
  };

  class TrainingNeuralNetwork: public NeuralNetwork {
  public:
    void addLayer(Layer *layer) override;

    void train(float *input, float *desiredOutput, float learningRate);
  };
}
