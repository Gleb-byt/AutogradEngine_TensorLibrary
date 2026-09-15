#include "tensor.hpp"

#include <vector>
#include <memory>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <string>

void save(const std::unordered_map<std::string, std::shared_ptr<Tensor>> & state_dict,
    const std::string & filenme);

std::unordered_map<std::string, std::shared_ptr<Tensor>> load(const std::string & filename);