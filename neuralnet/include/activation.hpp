#pragma once

namespace nnet {
  class ActivationFunction {
  public:
    virtual float activate(float input) {
      return 0;
    }

    virtual float derivative(float input) {
      return 0;
    }

    virtual float derivativeFromOutput(float output) {
      return 0;
    }
  };

  class SigmoidActivationFunction : public ActivationFunction {
    float activate(float input) override;

    float derivative(float input) override;
  
    float derivativeFromOutput(float output) override;
  };

  class RELUActivationFunction : public ActivationFunction {
    float activate(float input) override;

    float derivative(float input) override;
  
    float derivativeFromOutput(float output) override;
  };
}
