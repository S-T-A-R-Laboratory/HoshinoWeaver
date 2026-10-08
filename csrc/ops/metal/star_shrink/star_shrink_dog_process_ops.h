#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_star_shrink_dog_process_metal_ops(py::module_& m);
