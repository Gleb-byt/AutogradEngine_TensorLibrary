#include "../include/dataset.hpp"
#include "../include/tensor.hpp"

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

std::vector<std::vector<std::vector<float>>> read_mnist(std::string path) {


}


MNIST::MNIST(std::string data_path, std::string labels_path) {
    _images = read_mnist(data_path);
    _labels = read_mnist_labels(labels_path);
}