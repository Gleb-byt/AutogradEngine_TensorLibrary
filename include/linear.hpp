#pragma once
#include "module.hpp"
#include "tensor.hpp"

class Linear : public Module {

private:
    std::shared_ptr<Tensor> _weight;
    std::shared_ptr<Tensor> _bias;
    int _in_features;
    int _out_features;
    int _seed;

public:
    Linear(int in_features, int out_features, int seed);

    void reset_parameters();

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;

};