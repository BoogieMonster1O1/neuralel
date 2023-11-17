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

  TrainingLayer::TrainingLayer(int inputDimension, int outputDimension, ActivationFunction* act)
    : Layer(inputDimension, outputDimension, act) {
    this->input = new float[inputDimension];
    this->weightedInput = new float[outputDimension];
    this->output = new float[outputDimension];
    this->weightGradients = new float[inputDimension * outputDimension];
    this->biasGradients = new float[outputDimension];
  }

  TrainingLayer::~TrainingLayer() {
    delete[] input;
    delete[] weightedInput;
    delete[] output;
    delete[] weightGradients;
    delete[] biasGradients;
  }

  float* TrainingLayer::forward(float* input) {
    std::memcpy(this->input, input, sizeof(float) * this->inputDimension);
    
    float* result = new float[this->outputDimension];
    for (int i = 0; i < this->outputDimension; ++i) {
      result[i] = 0;
      for (int j = 0; j < this->inputDimension; ++j) {
	result[i] += input[j] * this->weights[i * this->inputDimension + j];
      }
      result[i] += this->biases[i];
    }

    std::memcpy(this->weightedInput, result, sizeof(float) * this->outputDimension);

    for (int i = 0; i < this->outputDimension; ++i) {
      result[i] = this->activationFunction->activate(result[i]);
    }

    std::memcpy(this->output, result, sizeof(float) * this->outputDimension);

    return result;
  }

  void TrainingLayer::updateWeightsAndBiases(float learningRate) {
    for (int i = 0; i < outputDimension; ++i) {
      for (int j = 0; j < inputDimension; ++j) {
	weights[i * inputDimension + j] -= learningRate * weightGradients[i * inputDimension + j];
      }
    }

    for (int i = 0; i < outputDimension; ++i) {
      biases[i] -= learningRate * biasGradients[i];
    }
  }

  void TrainingLayer::backward(float* targetOutput) {
    float* outputGradients = new float[outputDimension];
    for (int i = 0; i < outputDimension; ++i) {
      outputGradients[i] = 2.0 * (this->output[i] - targetOutput[i]);
    }

    for (int i = 0; i < outputDimension; ++i) {
      biasGradients[i] = outputGradients[i] * activationFunction->derivative(weightedInput[i]);
    }

    for (int i = 0; i < outputDimension; ++i) {
      for (int j = 0; j < inputDimension; ++j) {
	weightGradients[i * inputDimension + j] = outputGradients[i] * activationFunction->derivative(weightedInput[i]) * input[j];
      }
    }

    delete[] outputGradients;
  }
}
