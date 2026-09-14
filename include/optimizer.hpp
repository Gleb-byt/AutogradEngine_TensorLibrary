#pragma once

#include <memory>
#include <tensor.hpp>
#include <vector>


class Optimizer {
protected:
    std::vector<std::shared_ptr<Tensor>> _parameters;
    float _lr;

public:

    Optimizer(std::vector<std::shared_ptr<Tensor>> parameters, float lr);

    virtual void step() = 0;

    virtual ~Optimizer() = default;

    void zero_grad();


};

class SGD : public Optimizer {
private:
    float _momentum;
    std::vector<std::vector<float>> _velocities;


public:
    SGD(std::vector<std::shared_ptr<Tensor>> parameters, 
        float lr, float momentum = 0.0f
    );
    void step() override;


};