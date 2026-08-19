#pragma once
#include "tensor.hpp"
#include "module.hpp"
#include "autograd.hpp"
#include <functional>
#include <memory>


class Relu : public Module {
public:

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};