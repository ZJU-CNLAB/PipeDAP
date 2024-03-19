#include <torch/extension.h>
#include <pybind11/functional.h>

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "communicator.h"

namespace py = pybind11;


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init", &g_init, "init");
    m.def("rank", &g_rank, "rank");
    m.def("size", &g_size, "size");
    m.def("barriar", &g_barriar, "barriar");

    std::string name = std::string("Communicator");
    py::class_<Communicator>(m, name.c_str())
        .def(py::init<int>())
        .def("destroy", &Communicator::destroy)
        .def("reload", &Communicator::reload)
        .def("bcast", &Communicator::bcast)
        .def("reduce", &Communicator::reduce)
        .def("allReduce", &Communicator::allReduce)
        .def("allReduceRB", &Communicator::allReduceRB)
        .def("allReduceRSAG", &Communicator::allReduceRSAG)
        .def("reduceScatter", &Communicator::reduceScatter)
        .def("allGather", &Communicator::allGather)
        .def("multiBcast", &Communicator::multiBcast)
        .def("sendrecv", &Communicator::sendrecv)
        .def("synchronize", &Communicator::synchronize)
        .def("barrier", &Communicator::barrier)
        .def("syncStream", &Communicator::syncStream)
        .def("getNumOfFreeStreams", &Communicator::getNumOfFreeStreams)
        .def("__repr__", [](const Communicator &a) { return "Communicator"; });

}
