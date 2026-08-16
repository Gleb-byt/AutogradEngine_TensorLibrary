#include "../include/autograd.hpp"
#include "../include/tensor.hpp"
#include <memory>
#include <vector>


MatMulBackward::MatMulBackward(std::shared_ptr<Tensor> a_, 
    std::shared_ptr<Tensor> b_, 
    std::shared_ptr<Tensor> out_grad_) {
        this->a_ = a_;
        this->b_ = b_;
        this->out_grad_ = out_grad_->grad_;
    }

void MatMulBackward::apply() {
    if (a_->requires_grad_) {

        Tensor grad_a = (*out_grad_).matmul((*b_).transpose());
        if (!a_->grad_) {
            (*a_).grad_ = std::make_shared<Tensor>(grad_a);

        } else {
            *(*a_).grad_ = (*(*a_).grad_) + grad_a;
        }
    }
    if (b_->requires_grad_) {

        Tensor grad_b = (*a_).transpose().matmul(*out_grad_);
        if (!b_->grad_) {
            b_->grad_ = std::make_shared<Tensor>(grad_b);
        } else {
            *(*b_).grad_ = *(*b_).grad_ + grad_b;
        }
    }
}



/*
    Add functoin to unite graph parts
    For example Residual Block
*/ 


std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(*a + *b);
    result->requires_grad_ = a->requires_grad_ || b->requires_grad_;

    // for (int i {}; i < (*a).size(); ++i) {
    //     (*result)[i] = (*a)[i] + (*b)[i];
    // }

    if (result->requires_grad_) {
        result->grad_ = std::make_shared<Tensor>(result->shape_);
        result->grad_fn_ = std::make_shared<AddBackward>(a,b, result);
    }

    return result;
}   


std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    std::shared_ptr<Tensor> result = std::make_shared<Tensor> (*a * *b);
    result->requires_grad_ = a->requires_grad_ || b->requires_grad_;

    if (result-> requires_grad_) {
        result -> grad_ = std::make_shared<Tensor>(result->shape_);
        result->grad_fn_ = std::make_shared<MulBackward>(a,b, result);
    }
    return result;
}
