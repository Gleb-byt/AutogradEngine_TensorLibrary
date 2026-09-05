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
        .def(py::init<const std::vector<int> &, bool> (),
            py::arg("shape"), py::arg("requires_grad"),
            "Method used to create Tensor filled with default values, which requires grad"
        
        )
        .def("backward", &Tensor::backward)

        .def("zero_grad", & Tensor::zero_grad)

        .def("size", & Tensor::size)

        .def_readwrite("requires_grad", & Tensor::requires_grad_)

        .def_readonly("shape", & Tensor::shape_)

        .def("get_data", [] (const Tensor & t) -> std::vector<float> {
            return *(t.data_);
        }, "Return Tensor data as a python list")

        .def("get_grad", [] (const Tensor & t) -> std::vector<float> {
            if (t.requires_grad_) return *(*(t.grad_)).data_;
            return std::vector<float> ();
        }, "Returns data about Tensor gradient")

        .def("print", &Tensor::print);

    m.def("matmul", [] (const Tensor & a, const Tensor & b) -> Tensor {
        try {
            return a.matmul(b);
        }
        catch ( const std::string & error_message) {
            std::cout << error_message << std::endl;
        }
        return Tensor({1});
    }, py::arg("a"), py::arg("b"), 
    "Allows to multiply any Tensors no using inner methods"
    );

    m.def("add", [] (const Tensor & a, const Tensor & b) -> Tensor {
        try {
            return a + b;
        }
        catch (const std::string & error_message) {
            std::cout << error_message << std::endl;
        }
        return Tensor({1});
    }, py::arg("a"), py::arg("b"),
    "Allows to add any Tensors no using inner mehtods");

    m.def("mul", [] (const Tensor & a, const Tensor & b) -> Tensor {
        try {
            return a.dot(b);
        }
        catch (const std::string & error_message) {
            std::cout << error_message << std::endl;
        }

        return Tensor({1});
    }, py::arg("a"), py::arg("b"), 
    "Allows to multiply Tensors (val on val) no using inner methods"
    );



    py::class_<Linear, std::shared_ptr<Linear>> (m, "Linear")
        .def(py::init<int , int, int> (),
            py::arg("in_features"), py::arg("out_features"), py::arg("seed"),
            "This method is used to create Linear layer with input size of Tensor and output"
        )
        .def("forward", &Linear::forward)
        .def("__call__", &Linear::forward)
        .def("parameters", &Linear::parameters)
        .def("reset_parameters", &Linear::reset_parameters);


    py::class_<Relu, std::shared_ptr<Relu>> (m, "Relu")
        .def(py::init<> ())
        .def("forward", &Relu::forward, py::arg("input"))
        .def("__call__", &Relu::forward, py::arg("input"));

    py::class_<Flatten, std::shared_ptr<Flatten>> (m, "Flatten")
        .def(py::init<>())
        .def("forward", &Flatten::forward, py::arg("input"))
        .def("__cal__", &Flatten::forward, py::arg("input"));

    py::class_<MSELoss, std::shared_ptr<MSELoss>> (m, "MSELoss")
        .def(py::init<>())
        .def("forward", &MSELoss::forward, py::arg("pred"), py::arg("target"))
        .def("__call__", &MSELoss::forward, py::arg("pred"), py::arg("target"));

        
    py::class_<CrossEntropyLoss, std::shared_ptr<CrossEntropyLoss>> (m, "CrossEntropyLoss")
        .def(py::init<>())
        .def("forward", &CrossEntropyLoss::forward, py::arg("pred"), py::arg("target"))
        .def("__call__", &CrossEntropyLoss::forward, py::arg("pred"), py::arg("target"));
        
    
};