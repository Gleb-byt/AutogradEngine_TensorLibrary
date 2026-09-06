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


class model:
    def __init__(self, input_dimensions: int, output_dimensions: int, seed: int = 42):
        self.module1 = ae.Linear(input_dimensions, 512)
        self.module2 = ae.Linear(512, 256)
        self.module3 = ae.linear(256, 64)
        self.module4 = ae.Linear(16, output_dimensions)

    def make_pred(input_data: ae.Tensor): ...
