#pragma once
#include "../include/tensor.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class Module {

private:
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> _parameters;
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> _modules;

public:
    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input);
    std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> input);

    void register_parameter(std::string name, std::shared_ptr<Tensor> param);

    void register_module(std::string name, std::shared_ptr<Module> module);
};