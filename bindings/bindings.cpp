#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

#include "../include/autograd.hpp"
#include "../include/flatten.hpp"
#include "../include/linear.hpp"
#include "../include/loss_functions.hpp"
#include "../include/module.hpp"
#include "../include/relu.hpp"
#include "../include/tensor.hpp"

namespace py = pybind11;

PYBIND11_MODULE(autograd_engine, m) {
    m.doc() = "C++ autograd library based on Tensor operations";

    /*
    Class should be initialized with second
    argument which is used as an instrution for 
    memory control. Otherwise all program will fall with a
    misstake because it will breake the logic
    with pointers.
    */

    py::class_<Tensor,std::shared_ptr<Tensor>> (m, "Tensor")
        .def(py::init<const std::vector<int> &>(),
        py::arg("shape"),
        "Method used to create Tensor filled by default value"    
        ) 
        .def(py::init<const std::vector<float> &, const std::vector<int> & > (),
        py::arg("data"), py::arg("shape"),
        "Method used to create Tensor filled with input data"
        )
        .def("print", &Tensor::print);

}