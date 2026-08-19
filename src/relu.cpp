#include "../include/relu.hpp"
#include "../include/autograd.hpp"


std::shared_ptr<Tensor> Relu::forward(std::shared_ptr<Tensor> input) {
    std::shared_ptr<Tensor> result = std::make_shared<Tensor> (input->shape_);

    for (int i {}; i < input->size(); ++i) {
        float val = (*input)[i];
        (*result)[i] = val > 0.0f ? val : 0.0f;
    }

    result->requires_grad_ = input->requires_grad_;
    if (result->requires_grad_) {
        result->grad_ = std::make_shared<Tensor> (result->shape_);

        result->grad_fn_ = std::make_shared<ReluBackward> (input, result);
    }

    return result;
}
