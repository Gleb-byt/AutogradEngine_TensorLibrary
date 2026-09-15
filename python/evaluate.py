import build.bindings.autograd_engine as ae
from .nn import NeuralNetwork
import os

PATH: tuple[str, str] = (
    "data/Mnist/t10k-images-idx3-ubyte",
    "data/MNIST/t10k-labels-idx1-ubyte",
)

MODEL_PATH: str = "models/mnist_model.bin"


def evaluate(model_path: str, test_path_data: str, test_path_labels: str):
    mnist_test: ae.MNIST = ae.MNIST(test_path_data, test_path_labels)

    weights = ae.load(MODEL_PATH)

    model: NeuralNetwork = NeuralNetwork()

    model.load_state_dict(weights)

    batch_size = 1

    test_dataloader: ae.DataLoader = ae.DataLoader(mnist_test, batch_size, False)

    correct = 0

    made_predictions = 0

    am_samples = test_dataloader.n_samples()

    for batch in test_dataloader:
        for label, img in batch:
            pred = model.forward(img).get_data()

            pred = max(range(len(pred)), key=lambda x: pred[x])  # type: ignore

            if pred == label:
                correct += 1

            made_predictions += 1

        os.system("cls" if os.name == "nt" else "clear")

        print(f"{made_predictions} / {am_samples}")

    accuracy = correct / test_dataloader.n_samples() * 100
    print(f"accuracy = {accuracy:.2f} %")


evaluate(MODEL_PATH, PATH[0], PATH[1])
