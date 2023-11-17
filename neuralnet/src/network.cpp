#include "network.hpp"

namespace nnet {
  float* NeuralNetwork::forward(float* input) {
    float* output = input;

    for (auto i = this->layers.begin(); i != this->layers.end(); i++) {
      float* layerOutput = (*i)->forward(output);
      delete[] output;
      output = layerOutput;
    }

    return output;
  }

  void NeuralNetwork::addLayer(Layer* layer) {
    this->layers.push_back(layer);
  }

  void TrainingNeuralNetwork::addLayer(Layer* layer) {
    if (dynamic_cast<TrainingLayer*>(layer) == nullptr) {
      throw std::invalid_argument("Only TrainingLayer instances are allowed in TrainingNeuralNetwork!");
    }

    NeuralNetwork::addLayer(layer);
  }

  void TrainingNeuralNetwork::train(float* input, float* desiredOutput, float learningRate) {
    float* currentInput = input;
    
    for (auto it = layers.begin(); it != layers.end(); ++it) {
      auto currentLayer = dynamic_cast<TrainingLayer*>(*it);
      (*it)->forward(currentInput);
      if (std::next(it) != layers.end()) {
	auto nextLayer = dynamic_cast<TrainingLayer*>(*std::next(it));
	if (nextLayer) {
	  currentInput = currentLayer->output;
	}
      }
    }

    float* target = desiredOutput;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
      auto currentLayer = dynamic_cast<TrainingLayer*>(*it);
      if (currentLayer) {
	currentLayer->backward(target);
	target = currentLayer->input;
      }
    }

    for (auto it = layers.begin(); it != layers.end(); ++it) {
      auto currentLayer = dynamic_cast<TrainingLayer*>(*it);
      if (currentLayer) {
	currentLayer->updateWeightsAndBiases(learningRate);
      }
    }
  }
}
