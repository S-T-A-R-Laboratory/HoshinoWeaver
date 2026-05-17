#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_noise_ops(py::module_& m);
