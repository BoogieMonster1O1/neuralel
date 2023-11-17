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

  class TrainingLayer: public Layer {
  public:
    float* input;
    float* weighedInput;
    float* output;
    float* weightGradients;
    float* biasGradients;

    TrainingLayer(int inputDimension, int outputDimension, ActivationFunction* act);

    ~TrainingLayer();

    float* forward(float* input) override;
    
    void updateWeightsAndBiases(float learningRate);

    void backward(float* predictedOutput, float* targetOutput);
  };
}
