#include "../include/tensor.hpp"
#include "../include/flatten.hpp"
#include "../include/autograd.hpp"
#include <functional>
#include <memory>

std::shared_ptr<Tensor> Flatten::forward(std::shared_ptr<Tensor> input) {
    
    std::vector<int> new_shape = {input->size()};

    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(*(input->data_), new_shape);

    result->requires_grad_ = input->requires_grad_;

    if (result->requires_grad_) {
        result->grad_ = std::make_shared<Tensor>(result->shape_);
        result->grad_fn_ = std::make_shared<FlattenBackward>(input, result);
    }
    return result;


}
