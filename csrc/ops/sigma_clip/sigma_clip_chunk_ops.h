#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_sigma_clip_chunk_ops(py::module_& m);
