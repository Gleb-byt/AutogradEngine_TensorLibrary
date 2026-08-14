#pragma once
#include <vector>
#include <memory>
#include <stdexcept>
#include <numeric>
#include <iostream>

class Tensor {
private:

    std::shared_ptr<std::vector<float>> data_;
        
    std::vector<int> strides_;
    
public:
    std::vector<int> shape_;

    Tensor(const std::vector<int>& shape);

    Tensor(const std::vector<float>& data, const std::vector<int>& shape);

    int size() const;

    void compute_strides();

    float& operator[](int index) { return (*data_)[index]; }
    const float& operator[](int index) const {return (*data_)[index];}

    Tensor operator+(const Tensor&other) const;

    Tensor operator-(const Tensor&other) const;

    Tensor operator*(const Tensor&other) const;

    Tensor operator/(const Tensor&other) const;

    Tensor matmul(const Tensor&other) const;

    void print() const;




};