import build.bindings.autograd_engine as ae
import python.nn as nn

# for debug mode code should be executed using next line:
# py -3.13 -X faulthandler -m python.train
# faulthandler allows to catch C++ misstakes of OS and
# reflect them into terminal


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

    total_batch_amount: int = len(train_dataloader)

    print("dataloader initialized")

    model: ae.Module = nn.NeuralNetwork()
    loss_fn: ae.Module = ae.CrossEntropyLoss()

    params = [p for _, p in model.parameters()]

    learning_rate = 0.001

    optimizer = ae.SGD(params, learning_rate, 0.9)

    n_epochs: int = 5

    print("hyper parameters")

    for epoch in range(3):

        total_loss = 0
        batch_am = 0
        for batch in train_dataloader:
            batch_am += 1
            for target_label, img_tensor in batch:
                target_tensor = ae.Tensor([float(target_label)], [1])

                pred = model.forward(img_tensor).reshape([1, 10])

                loss: ae.Tensor = loss_fn.forward(pred, target_tensor)

                total_loss += loss.get_data()[0]

                loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            ae.save(model.state_dict(), "models/mnist_model.bin")

            print(f"Loss = {loss.get_data()[0]}; batch {batch_am} / {total_batch_amount}")  # type: ignore

        print(
            f"Epoch {epoch + 1}/{n_epochs}, Loss = {total_loss / train_dataloader.n_samples()}"
        )


if __name__ == "__main__":
    train_new_mnist_model()
