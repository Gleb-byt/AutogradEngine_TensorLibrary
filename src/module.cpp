#include "../include/module.hpp"
#include "../include/tensor.hpp"
#include <iostream>
#include<memory>
#include<utility>
#include<vector>
#include<stdexcept>

std::shared_ptr<Tensor> Module::forward(std::shared_ptr<Tensor> input) {
    throw std::runtime_error("Forward not implemented");
}

std::shared_ptr<Tensor> Module::operator()(std::shared_ptr<Tensor> input) { return forward(input) };


void Module::register_parameter(std::string name, std::shared_ptr<Tensor> param) {
    for (const auto &p: _parameters) {
        if (p.first == name) {
            throw std::runtime_error("Parameter" + name + "already registered");
        }
    }

    _parameters.push_back({name, param});
}

void Module::register_module(std::string name, std::shared_ptr<Module> module) {
    for ( const auto &m : _modules ) {
        if (m.first == name) {
            throw std::runtime_error("Module" + name + "already registered");
        }
    }

    _modules.push_back({name, module});
}