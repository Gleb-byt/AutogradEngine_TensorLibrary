#include "../include/dataloader.hpp"
#include "../include/dataset.hpp"



int main() {

    MNIST mnist_train = 
        MNIST("./data/train-images-idx3-ubyte", "./data/train-labels-idx1-ubyte");

    std::cout << "Datasets succesfully loaded" << std::endl;

    int batch_size = 10;

    DataLoader mnist_train_loader(& mnist_train, batch_size, true);

    std::cout << "Visualizing first batch of training data" << std::endl;

    for (auto batch : mnist_train_loader) {
        for (auto item : batch) {
            visualize_image(item.second);
            std::cout << mnist_train.label_to_class(item.first) << std::endl;
        }
        break;
    }

    return 0;
}

/*
    Tests should be launched from 
    the project directory, otherwise 
    datasets path will be invalid
*/

/*
    After launching tests you should be
    able to recognize a dress or a T-shirt.
    If it is true therefore everything works correct
*/