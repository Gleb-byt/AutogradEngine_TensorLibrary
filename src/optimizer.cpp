#include "../include/tensor.hpp"
#include "../include/optimizer.hpp"

#include <memory>
#include <vector>


void Optimizer::zero_grad() {
    for (std::shared_ptr<Tensor> param : _parameters) {
        if (param) {
            param->zero_grad();
        }
    }
}

Optimizer::Optimizer(std::vector<std::shared_ptr<Tensor>> parameters, float lr) :
    _parameters(parameters),
    _lr(lr) {}


/*
    Where
    v_t = new_weight ; y = impuls coefficient(momentum)
    g_t = derivative of current batch

    n = learning rate

    formula: v_t = y * v_{t-1} + g_t

    w_t = w_{t-1} - n *  v_t
*/


SGD::SGD(std::vector<std::shared_ptr<Tensor>> parameters, 
    float lr, float momentum
) :
    Optimizer(parameters,lr),
    _momentum(momentum) {

    if (_momentum > 0.0f) {
        _velocities.resize(parameters.size());
        for (size_t i {}; i < _parameters.size(); ++i) {
            if (_parameters[i]) {
                    _velocities[i].resize(_parameters[i]->size(), 0.0f);

            }
        }
    }

}

void SGD::step() {
    for (size_t p_idx {}; p_idx < _parameters.size(); ++p_idx) {
        auto & param = _parameters[p_idx];

        if (!param || !param->requires_grad_ || !param->grad_) {
            continue;
        }

        auto & param_data = *(param->data_);
        auto & grad_data = *(param->grad_->data_);
        size_t n = param->size();

        if (_momentum > 0.0f) {
            auto & v = _velocities[p_idx];

            for (size_t i {}; i < n; ++i) {
                v[i] = _momentum * v[i] + grad_data[i];

                param_data[i] -= _lr * v[i];
            }

        } else {
            for (size_t i {}; i < n; ++i) {
                param_data[i] -= _lr * grad_data[i];
            }
        }
    }
}