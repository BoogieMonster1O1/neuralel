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
    delete weights;
    delete biases;
  }
}
