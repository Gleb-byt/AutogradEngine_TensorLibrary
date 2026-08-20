#pragma once
#include "module.hpp"
#include "tensor.hpp"
#include <memory>
#include <cmath>

class MSELoss : public Module {
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target);
}
;

class CrossEntropyLoss : public Module {
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target);
};

