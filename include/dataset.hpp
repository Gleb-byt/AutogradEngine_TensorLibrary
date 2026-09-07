#pragma once

#include "tensor.hpp"
#include <memory>
#include <string>
#include <utility>
#include <vector>

class Dataset {
public:
    virtual std::pair<int, std::shared_ptr<Tensor>> get_item(int index)  = 0;
    virtual int get_legth() = 0;


};

class MNIST : public Dataset {
private:
    std::vector<std::vector<std::vector<float>>> _images;
    std::vector<int> _labels;
    std::vector<std::string> classes = {"zero", "one", "two", "three", "four",
        "five", "size", "seven", "eight", "nine"    
    };

public:
    MNIST(std::string data_path, std::string labels_path);
};