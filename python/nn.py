# before applying any python code
# an essential thin is to compile
# CMakeLists.txt file using next commands:
# cmake build ..
# cmake --build build
# in git i will save the version which
# already includes compilled .pyd file in
# python folder


import sys

print(sys.version)

import build.bindings.autograd_engine as ae

shape = [2, 2]
data = list(map(float, range(1, 5)))

t = ae.Tensor(data=data, shape=shape)

print(data)
t.print()
