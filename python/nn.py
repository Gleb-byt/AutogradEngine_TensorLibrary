# before applying any python code
# an essential thin is to compile
# CMakeLists.txt file using next commands:
# cmake build ..
# cmake --build build
# in git i will save the version which
# already includes compilled .pyd file in
# python folder


# import sys

# print(sys.version)

# import build.bindings.autograd_engine as ae

# shape = [2, 2]
# data = list(map(float, range(1, 5)))

# t = ae.Tensor(data=data, shape=shape)

# print(data)
# t.print()

import typing as tp

import build.bindings.autograd_engine as ae


class NeuralNetwork(ae.Module):
    def __init__(self):
        self._flatten = ae.Flatten()
        self._linear1 = ae.Linear(28 * 28, 512)
        self._linear2 = ae.Linear(512, 512)
        self._linear3 = ae.Linear(512, 10)
        self._relu = ae.Relu()

        self.register_module("linear_1", self._linear1)
        self.register_module("linear_2", self._linear2)
        self.register_module("linear_3", self._linear3)

    def forward(self, input: ae.Tensor):
        flat = self._flatten(input)
        linear_1 = self._lienar1(flat)
        relu_1 = self._relu(linear_1)
        linear_2 = self._linear2(relu_1)
        relu_2 = self._relu(linear_2)
        return self._linear3(relu_2)
