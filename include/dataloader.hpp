#pragma once
#include "dataset.hpp"
#include <memory>
#include <utility>
#include <vector>
#include <algorithm>
#include <random>

class DataLoader {
private:
    Dataset * _dataset;
    int _batch_size;
    std::vector<int> _indices;

public:
    DataLoader(Dataset * dataset, int batch_size, bool shuffle);

    class Iterator {
    private:
        DataLoader * _dataloader;
        int _index;
    public:
        Iterator(DataLoader *dataloader, int index);
        void operator++();
        std::vector<std::pair<int, std::shared_ptr<Tensor>>> operator*();
    };
};