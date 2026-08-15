#pragma once
#include <vector>
#include <memory>
#include "tensor.hpp"

class BackwardFunction {
public:
    virtual ~BackwardFunction() = default;
    
    virtual void apply() = 0;
};

class AddBackward : public BackwardFunction {

private:
    std::shared_ptr<Tensor> a_;
    std::shared_ptr<Tensor> b_;
    std::shared_ptr<Tensor> out_grad_;

public:
    AddBackward(std::shared_ptr<Tensor> a,
        std::shared_ptr<Tensor> b, 
        std::shared_ptr<Tensor> out) {
            this->a_ = a;
            this->b_ = b;
            this->out_grad_ = out->grad_;
        }

    void apply() override {

        if (a_->requires_grad_) {
            if (!a_->grad_) {
                (*a_).grad_ = std::make_shared<Tensor>((*a_).shape_);
            }
            for (int i {}; i < (*a_).size(); ++i) {
                (*(*a_).grad_)[i] += (*out_grad_)[i];
            }
        }

        if (b_->requires_grad_) {
            if (!b_->grad_) {
                (*b_).grad_ = std::make_shared<Tensor>((*b_).shape_);
            }
            for (int i {}; i < (*b_).size(); ++i) {
                (*b_->grad_)[i] += (*out_grad_)[i];
            }
        }

    }
};

class MulBackward : public BackwardFunction {

private:
    std::shared_ptr<Tensor> a_;
    std::shared_ptr<Tensor> b_;
    std::shared_ptr<Tensor> out_grad_;

public:
    MulBackward(std::shared_ptr<Tensor> a_,
        std::shared_ptr<Tensor> b_,
        std::shared_ptr<Tensor> out_grad_) {
            this->a_ = a_;
            this->b_ = b_;
            this->out_grad_ = out_grad_;
        }

    void apply() override {
        if (a_->requires_grad_) {
            if (!(*a_).grad_) {
                (*a_).grad_ = std::make_shared<Tensor>((*a_).shape_);
            }
            for (int i {}; i < (*a_).size(); ++i) {
                (*a_->grad_)[i] += (*out_grad_)[i] * (*b_)[i];
            }
        }
        if ((*b_).requires_grad_) {
            if (!b_->grad_) {
                (*b_).grad_ = std::make_shared<Tensor>((*b_).shape_);
            }
            for (int i {}; i < b_->size(); ++i) {
                (*(*b_).grad_)[i] += (*out_grad_)[i] * (*a_)[i];
            }
        }
    }

};