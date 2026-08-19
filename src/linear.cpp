#include "../include/linear.hpp"
#include "../include/tensor.hpp"
#include "../include/autograd.hpp"
#include <memory>
#include <random>
#include <cmath>

Linear::Linear(int in_features, int out_features, int seed) : 
    _in_features(in_features),
    _out_features(out_features),
    _weight(std::make_shared<Tensor> (std::vector<int> {in_features, out_features}, true)),
    _bias(std::make_shared<Tensor> (std::vector<int> {out_features}, true)),
    _seed(seed)
{
    register_parameter("weight", _weight);
    register_parameter("bias", _bias);

    reset_parameters();
};

void Linear::reset_parameters() {
    float gain = std::sqrt(2.0f);
    std::size_t fan_in = _in_features;
    float bound = gain * std::sqrt(3.0f / fan_in);
    std::mt19937 generator(_seed);

    for (std::size_t i = 0; i < _weight->size(); ++i) {
        (*_weight)[i] = std::uniform_real_distribution<float>(-bound, bound)(generator);
    }
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> input) {
    std::shared_ptr<Tensor> xW = matmul(input, _weight);

    return add(xW, _bias);
}