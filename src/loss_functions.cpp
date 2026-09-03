#include "../include/loss_functions.hpp"
#include "../include/autograd.hpp"
#include "../include/tensor.hpp"


#include <memory>
#include <vector>
#include <cmath>
#include <algorithm>




std::shared_ptr<Tensor> MSELoss::forward(
    std::shared_ptr<Tensor> pred,
    std::shared_ptr<Tensor> target
) {
    if (pred->shape_ != target->shape_) {
        throw std::invalid_argument("Shape of predicted Tensor should be equal to the  \
            shape of target Tensor");
    }


    float res = 0.0f;

    for (int i {}; i < pred->size(); ++i) {
        float diff = (*pred)[i] - (*target)[i];
        res += diff * diff;
    }

    std::shared_ptr result = std::make_shared<Tensor>(std::vector<int>{1});

    (*result)[0] = res / pred->size();


    result->requires_grad_ = pred->requires_grad_;
    if (pred->requires_grad_) {
        result->grad_ = std::make_shared<Tensor>(result->shape_);
        result->grad_fn_ = std::make_shared<BackwardFunction> (pred, target, result);
    }

    return result;
}


std::shared_ptr<Tensor> CrossEntropyLoss::forward(
    std::shared_ptr<Tensor> pred,
    std::shared_ptr<Tensor> target
) {
    if (pred->shape_.size() != 2 || target->shape_.size()!=1) {
        throw std::invalid_argument("CE requires input of shape [N, M], and output [N]");
    }

    int batch_size = pred->shape_[0];

    int num_classes = pred->shape_[1];

    std::vector<float> softmax_cache_(pred->size(), 0.0f);

    float loss {};

    for (int i {}; i < batch_size; ++i) {
        float max_val = (*pred)[i * pred->strides_[0]];

        for (int j {}; j < num_classes; ++j) {
            max_val = std::max(max_val, (*pred)[i * pred->strides_[0] + j * pred->strides_[1]]);
        }

        float sum_exp {};

        for (int j {}; j < num_classes; ++j) {
            float e = std::exp((*pred)[i * num_classes + j] - max_val);
            softmax_cache_[i * pred->strides_[0] + j * pred->strides_[1]];
            sum_exp += e;
        }

        for (int j {}; j < num_classes; ++j) {
            softmax_cache_[i * (*pred).strides_[0] + j * (*pred).strides_[1]] /= sum_exp;

        }

        int true_pred = static_cast<int>((*target)[i]);

        loss += -std::log(softmax_cache_[i * num_classes + true_pred] + 1e-7);

    }

    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(std::vector<int> {1});
    (*result)[0] = loss / batch_size;

    result->requires_grad_ = pred->requires_grad_;

    if (result->requires_grad_) {
        result->grad_ = std::make_shared<Tensor>(result->shape_);
        result->grad_fn_ = std::make_shared<BackwardFunction>(pred, target, result, softmax_cache_);
    }

    return result;


}