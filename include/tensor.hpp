#pragma once
#include <vector>
#include <memory>
#include <stdexcept>
#include <numeric>
#include <iostream>


class BackwardFunction;

class Tensor {
private:

    std::vector<int> strides_;

    std::shared_ptr<std::vector<float>> data_;
    
public:
    bool requires_grad_ = false;

    std::shared_ptr<Tensor> grad_ = nullptr;

    std::shared_ptr<BackwardFunction> grad_fn_ = nullptr;

    std::vector<int> shape_;

    Tensor(const std::vector<int>& shape);

    Tensor(const std::vector<float>& data, const std::vector<int>& shape);

    Tensor(const std::vector<int>& shape, bool requires_grad);

    void zero_grad();

    int size() const;

    void compute_strides();

    float& operator[](int index) { return (*data_)[index]; }
    const float& operator[](int index) const {return (*data_)[index];}

    Tensor operator+(const Tensor&other) const;

    Tensor operator-(const Tensor&other) const;

    Tensor operator*(const Tensor&other) const;

    Tensor operator/(const Tensor&other) const;

    Tensor matmul(const Tensor&other) const;

    Tensor operator+(const float& val) const;

    Tensor operator*(const float& val) const;

    Tensor transpose() const;


    void print() const;




};