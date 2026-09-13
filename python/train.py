import build.bindings.autograd_engine as ae
import python.nn as nn


def train_new_mnist_model():
    print("Loading dataset")

    mnist_train: ae.MNIST = ae.MNIST(
        "data/Mnist/train-images-idx3-ubyte", "data/MNIST/train-labels-idx1-ubyte"
    )
    mnist_test: ae.MNIST = ae.MNIST(
        "data/Mnist/t10k-images-idx3-ubyte", "data/MNIST/t10k-labels-idx1-ubyte"
    )

    print("Dataset loaded")

    batch_size: int = 10

    train_dataloader: ae.DataLoader = ae.DataLoader(mnist_train, batch_size, True)

    test_dataloader: ae.DataLoader = ae.DataLoader(mnist_test, batch_size, False)

    model: ae.Module = nn.NeuralNetwork()
    loss_fn: ae.Module = ae.CrossEntropyLoss()

    learning_rate = 0.001
