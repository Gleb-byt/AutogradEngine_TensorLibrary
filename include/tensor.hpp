#pragma once
#include <vector>
#include <memory>
#include <stdexcept>
#include <numeric>
#include <iostream>


class BackwardFunction;

/*
    Importing "std::enable_shared_from_this" is needed to call
    such pointers in the function std::shared_ptr<Tensor>(this).
    The main problem why i can't do this is that this approach
    will create two links on the same object and one link after finishing
    its cycle will delete the main object which will cause undefined behaviour

    So you can apply this operation simply writing shared_from_this() in the
    function it is needed. 

    For example "build_topo(shared_from_this(), topo,visited)"
*/


class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    std::vector<int> strides_;

    std::shared_ptr<std::vector<float>> data_;

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

    Tensor dot(const Tensor& other) const;


    void print() const;

    void backward();




};