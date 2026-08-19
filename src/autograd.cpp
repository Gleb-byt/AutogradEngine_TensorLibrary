#include "../include/autograd.hpp"
#include "../include/tensor.hpp"
#include <memory>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <cstdlib>
#include <ctime>


MatMulBackward::MatMulBackward(std::shared_ptr<Tensor> a_, 
    std::shared_ptr<Tensor> b_, 
    std::shared_ptr<Tensor> out_grad_) {
        this->a_ = a_;
        this->b_ = b_;
        this->out_grad_ = out_grad_->grad_;
    }

void MatMulBackward::apply() {

    Tensor grad_a((*a_).shape_);
    Tensor grad_b((*b_).shape_);
    if (a_->shape_.size() == 1 && b_->shape_.size() == 2) {
        for (int i {}; i < a_->size();  ++i) {
            for (int j {}; j < b_->shape_[1]; ++j) {
                (*grad_a.data_)[i * grad_a.strides_[0]] += 
                (*(*b_).data_)[i * b_->strides_[0] + j * b_->strides_[1]] * 
                (*out_grad_->data_)[j * out_grad_->strides_[0] ];
            }
        }
        for (int i {}; i < b_->shape_[0]; ++i) {
            for (int j {}; j < b_->shape_[1]; ++j) {
                (*grad_b.data_)[i * grad_b.strides_[0] + j * grad_b.strides_[1]] = 
                (*out_grad_->data_)[j * out_grad_->strides_[0]] *
                (*a_->data_)[i * a_->strides_[0]];
            }
        }
        
    }


    if (a_->shape_.size() == 2 && b_->shape_.size() == 1) {
        for (int i {}; i < a_->shape_[0]; ++i) {
            for (int j {}; j < a_->shape_[1]; ++j) {
                (*grad_a.data_)[i * grad_a.strides_[0] + j * grad_a.strides_[1]] =
                (*out_grad_->data_)[i  * out_grad_->strides_[0]] *
                (*b_->data_)[j * b_->strides_[0]];
            }
        }
        for (int i {}; i < b_->size();  ++i) {
            for (int j {}; j < a_->shape_[0]; ++j) {
                (*grad_b.data_)[i * grad_b.strides_[0]] += 
                (*(*a_).data_)[j * a_->strides_[0] + i * a_->strides_[1]] * 
                (*out_grad_->data_)[j * out_grad_->strides_[0]];
            }
        }
        
    }


    if (a_->shape_.size() == 2 && b_->shape_.size() == 2) {
        grad_a = (*out_grad_).matmul((*b_).transpose());
        grad_b = (*a_).transpose().matmul(*out_grad_);

    }

    if (a_->shape_.size() == 1 && b_->shape_.size() == 1) {
        grad_a = (*b_) * (*out_grad_->data_)[0];
        grad_b = (*a_) * (*out_grad_->data_)[0];

    }

    if (a_->requires_grad_) {
        
        if (!a_->grad_) {
            (*a_).grad_ = std::make_shared<Tensor>(grad_a);

        } else {
            *(*a_).grad_ = (*(*a_).grad_) + grad_a;
        }
    }
    if (b_->requires_grad_) {

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


std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    std::shared_ptr<Tensor> result = std::make_shared<Tensor> ((*a).matmul(*b));

    result->requires_grad_ = a->requires_grad_ || b->requires_grad_;
    if (result -> requires_grad_) {
        result -> grad_ = std::make_shared<Tensor>(result->shape_);
        result->grad_fn_ = std::make_shared<MatMulBackward>(a,b,result);
    }

}


/*
    static in that position is used to show that this function is
    used only in this file and can't be used from any other place
*/  

/*
    visited set saves simple pointers not std::shared_ptr<T>
    so if it saves simple pointers to memory it doesn't need the
    structure with two pointers and smart logic
*/

static void build_topo(
    const std::shared_ptr<Tensor>& v,
    std::vector<std::shared_ptr<BackwardFunction>>& topo,
    std::unordered_set<BackwardFunction*>& visited
) {
    if (!v || !v->grad_fn_) return;

    BackwardFunction* fn_ptr = v->grad_fn_.get();
    if (visited.find(fn_ptr) == visited.end()) {
        visited.insert(fn_ptr);

        for (const auto& parent : v->grad_fn_->get_parents()) {
            build_topo(parent, topo, visited);
        }

        topo.push_back(v->grad_fn_);
    }
}

void Tensor::backward() {
    
    std::srand(std::time(NULL));

    if (!grad_) {
        grad_ = std::make_shared<Tensor>(shape_);
    }
    for (int i = 0; i < size(); ++i) {
        float val = static_cast<float>(std::rand()) / RAND_MAX;
        val = val == 0.0f ? 1e-7 : val;
        (*grad_)[i] = val;
    }

    std::vector<std::shared_ptr<BackwardFunction>> topo;
    std::unordered_set<BackwardFunction*> visited;

    build_topo(shared_from_this(), topo, visited);

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->apply();
    }
}

void Tensor::zero_grad() {
    if (grad_) {
        for (int i {}; i < size(); ++i) {
            (*grad_)[i] = 0.0f;
        }
    }
}


FlattenBackward::FlattenBackward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output) {
    this->input_ = input;
    this->out_ = output;
}

void FlattenBackward::apply() {
    if (input_->requires_grad_) {
        if (!input_->grad_) {
            input_->grad_ = std::make_shared<Tensor> (input_->shape_);
        }

        for (int i {}; i < input_->size(); ++i) {
            (*input_->grad_)[i] = (*out_->grad_)[i];
        }
    }

}

ReluBackward::ReluBackward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output) {
    this->input_ = input;
    this->out_ = output;
}

void ReluBackward::apply() {
    if (input_->requires_grad_) {
        if (!input_->grad_) {
            (*input_).grad_ = std::make_shared<Tensor> (input_->shape_);
        }

        for (int i {}; i < input_->size(); ++i) {
            if ((*input_)[i] > 0.0f) {
                (*(*input_).grad_)[i] += (*out_->grad_)[i];
            }
        }
    }
}