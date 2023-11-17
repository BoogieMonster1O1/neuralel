#pragma once

#include "activation.hpp"

namespace nnet {
  class Layer {
  public:
    float* weights;
    float* biases;
    ActivationFunction* activationFunction;
    int inputDimension;
    int outputDimension;

    Layer(int inputDimension, int outputDimension, ActivationFunction* act);

    ~Layer();

    float* forward(float* input);
  };
}
