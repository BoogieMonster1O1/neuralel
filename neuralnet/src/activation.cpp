#include "activation.hpp"
#include <cmath>

float nnet::SigmoidActivationFunction::activate(float input) {
  return 1.0f / (1.0f + std::exp(-input));
}

float nnet::SigmoidActivationFunction::derivative(float input) {
  double sig = activate(input);
  return sig * (1 - sig);
}

float nnet::SigmoidActivationFunction::derivativeFromOutput(float output) {
  return output * (1 - output);
}

float nnet::RELUActivationFunction::activate(float input) {
  return input > 0 ? input : 0;
}

float nnet::RELUActivationFunction::derivative(float input) {
  return input > 0 ? 1 : 0;
}

float nnet::RELUActivationFunction::derivativeFromOutput(float output) {
  return output != 0 ? 1 : 0;
}
