#pragma once
#include "module.hpp"
#include "tensor.hpp"
#include <memory>

class Flatten : public Module {
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};