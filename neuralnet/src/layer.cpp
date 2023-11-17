#include "layer.hpp"

namespace nnet {
  Layer::Layer(int inputDimension, int outputDimension, ActivationFunction* act) {
    this->inputDimension = inputDimension;
    this->outputDimension = outputDimension;
    this->activationFunction = act;
    this->weights = new float[inputDimension * outputDimension];
    this->biases = new float[outputDimension];
  }

  Layer::~Layer() {
    delete[] weights;
    delete[] biases;
  }

  float* Layer::forward(float* input) {
    float* result = new float[this->outputDimension];
    for (int i = 0; i < this->outputDimension; ++i) {
      result[i] = 0;
      for (int j = 0; j < this->inputDimension; ++j) {
	result[i] += input[j] * this->weights[i * this->inputDimension + j];
      }
      result[i] += this->biases[i];
    }

    for (int i = 0; i < this->outputDimension; ++i) {
      result[i] = this->activationFunction->activate(result[i]);
    }

    return result;
  }
}
