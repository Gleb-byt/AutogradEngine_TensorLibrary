#include "../include/tensor.hpp"
#include "../include/autograd.hpp"


Tensor::Tensor(const std::vector<int>&shape) : shape_(shape) {
    int total_size = 1;
    for (int dim : shape) {
        total_size *= dim;
    }
    data_ = std::make_shared<std::vector<float>>(total_size, 0.0f);
    compute_strides();   

}

Tensor::Tensor(const std::vector<float>& data, const std::vector<int>& shape):
    shape_(shape) {
        data_ = std::make_shared<std::vector<float>>(data);
        compute_strides();
    }

int Tensor::size() const {
    return data_->size();
    // or (*data_).size();
}

void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    int stride = 1;
    for (int i = shape_.size() - 1; i >= 0 ; --i) {
        strides_[i] = stride;
        stride *= shape_[i];
    }
}

Tensor Tensor::operator+(const Tensor&other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Shapes must match for addition");
    }
    Tensor result(shape_);
    for  (int i = 0; i < (*data_).size(); ++i) {
        result[i] = (*data_)[i] + (*other.data_)[i];
    }
    return result;
}


Tensor Tensor::operator-(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Shapes must match for subtraction");
    }
    Tensor result(shape_);
    for (int i {}; i < (*data_).size(); ++i) {
        result[i] = (*data_)[i] - (*other.data_)[i];
    }
    return result;
}


Tensor Tensor::operator*(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Shapes must match for multiplication");
    }
    Tensor result(shape_);
    for (int i {}; i < (*data_).size(); ++i) {
        result[i] = (*data_)[i] * (*other.data_)[i];
    }
    return result;

}

Tensor Tensor::operator/(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Shapes must match for devision");
    }
    float eps = 1e-6;
    Tensor result(shape_);
    for (int i {}; i < (*data_).size(); ++i) {
        result[i] = (*other.data_)[i] == 0.0 ? (*data_)[i] / eps : 
        (*data_)[i] / (*other.data_)[i];
    }
    return result;
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (shape_.size() == 1 && other.shape_.size() == 1) {

        return this->dot(other);
    }

    if (shape_.size() == 1 && other.shape_.size() == 2) {

        int K = shape_[0];

        if (K != other.shape_[0]) {
            throw std::invalid_argument("Inner Dimensions must match for 1D and 2D matmul");
        }

        int N = other.shape_[1];

        Tensor result({N});

        for (int i {}; i < N; ++i) {
            int sum {};
            for (int j {}; j < K; ++j) {
                sum += (*this->data_)[i] * (*other.data_)[i * other.strides_[0] + j * other.strides_[1]];
            }
            (*result.data_)[i] = sum;
        }
        return result;

    }

    if (shape_.size() == 2 && other.shape_.size() == 1) {
        int K = shape_[1];

        if (K != other.shape_[0]) {
            throw std::invalid_argument("Inner Dimensions must match for 2D and 1D matmul");
        }

        int N = other.shape_[1];

        Tensor result({N});

        for (int i {}; i < N; ++i) {
            int sum {};
            for (int j {}; j < K; ++j) {
                sum += (*this->data_)[i * this->strides_[0] + j * this->strides_[1]] * (*other.data_)[i + other.strides_[0]];
            }
            result[i] = sum;
        }
        return result;
    }

    if (shape_.size() == 2 && shape_.size() == 2) {

        int M = shape_[0];
        int K = shape_[1];
        int N = other.shape_[1];


        if (K != other.shape_[0]) {
            throw std::invalid_argument("Inner dimensions must match for 2D and 2D matmul");
        }

        Tensor result({M,N});

        for (int i {}; i < M; ++i) {
            for (int j {}; j < N; ++j) {
                float res = 0.0f;
                for (int k {}; k < K; ++k) {
                    res += (*data_)[i * strides_[0] + k * strides_[1]] *
                        (*other.data_)[k * other.strides_[0] + j * other.strides_[1]];
                }
                result[i * result.strides_[0] + j * result.strides_[1]] = res;
            }
        }
        return result;

    }

}

void Tensor::print() const {
    std::cout << "Tensor(";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i] << (i == shape_.size() - 1 ? "" : ", ");
    }
    std::cout << ")" << std::endl;
}



Tensor Tensor::operator*(const float& val) const {
    Tensor result_(shape_);
    for (int i {}; i < size(); ++i) {
        result_[i] = (*data_)[i] * val;
    }
    return result_;
}

Tensor Tensor::operator+(const float& val) const {
    Tensor result_(shape_);
    for  (int i {}; i < size(); ++ i) {
        result_[i] = (*data_)[i] + val;
    }
    return result_;
}


Tensor Tensor::transpose() const {
    if (shape_.size() != 2) {
        throw std::invalid_argument("Only 2D tensors could be transposed");
    } 
    Tensor result_({shape_[1], shape_[0]});
    for (int i {}; i < shape_[0]; ++i) {
        for (int j {}; j < shape_[1]; ++j) {
            result_[j * result_.strides_[0] + i * result_.strides_[1]] = (*data_)[i* strides_[0] + j * strides_[1]];
        }
    }
    return result_;

}


Tensor Tensor::dot(const Tensor& other) const {
    if (shape_.size() != 1 || other.shape_.size() != 1) {
        throw std::invalid_argument("This operation is allowed only for 1D Tensors");
    }

    if (shape_[0] != other.shape_[0]) {
        throw std::invalid_argument("Tensors should have the same length");
    }

    float sum = 0.0f;
    for (int i {}; i < size(); ++i) {
        sum += (*data_)[i] * (*other.data_)[i];
    }

    Tensor result({1});

    result[0] = sum;
    return result;
}