#include "../include/serialization.hpp"

const int MAGIC_NUMBER = 777;

void save(const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict,
          const std::string &filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    file.write(reinterpret_cast<const char *>(&MAGIC_NUMBER), sizeof(int));
    size_t num_tensors = state_dict.size();
    file.write(reinterpret_cast<const char *>(&num_tensors), sizeof(size_t));

    for (const auto &[weight_name, weight] : state_dict)
    {
        size_t name_len = weight_name.size();
        file.write(reinterpret_cast<const char *>(&name_len), sizeof(size_t));
        file.write(weight_name.data(), name_len);

        size_t shape_length = weight->shape_.size();
        file.write(reinterpret_cast<const char *>(&shape_length), sizeof(size_t));
        file.write(reinterpret_cast<const char *>(weight->shape_.data()),
                   shape_length * sizeof(int));

        size_t data_length = weight->size();
        file.write(reinterpret_cast<const char *>(&data_length), sizeof(size_t));
        file.write(reinterpret_cast<const char *>(weight->data_->data()),
                   data_length * sizeof(float));
    }
}

std::unordered_map<std::string, std::shared_ptr<Tensor>> load(const std::string &filename)
{
    std::unordered_map<std::string, std::shared_ptr<Tensor>> state_dict;
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open " + filename);
    }

    int magic = 0;
    file.read(reinterpret_cast<char *>(&magic), sizeof(int));
    if (magic != MAGIC_NUMBER)
    {
        throw std::runtime_error("Bad file format: wrong magic number");
    }

    size_t num_tensors = 0;
    file.read(reinterpret_cast<char *>(&num_tensors), sizeof(size_t));

    for (size_t t = 0; t < num_tensors; ++t)
    {
        size_t name_len = 0;
        file.read(reinterpret_cast<char *>(&name_len), sizeof(size_t));
        std::string weight_name(name_len, '\0');
        file.read(weight_name.data(), name_len);

        size_t shape_length = 0;
        file.read(reinterpret_cast<char *>(&shape_length), sizeof(size_t));
        std::vector<int> shape(shape_length);
        file.read(reinterpret_cast<char *>(shape.data()), shape_length * sizeof(int));

        size_t data_length = 0;
        file.read(reinterpret_cast<char *>(&data_length), sizeof(size_t));
        std::vector<float> raw(data_length);
        file.read(reinterpret_cast<char *>(raw.data()), data_length * sizeof(float));

        auto tensor = std::make_shared<Tensor>(raw, shape);
        tensor->requires_grad_ = true;

        state_dict[weight_name] = tensor;
    }

    return state_dict;
}