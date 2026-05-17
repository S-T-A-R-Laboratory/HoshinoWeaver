#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_camera_model_remap_fused_ops(py::module_& m);
