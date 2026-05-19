#pragma once

#include "../../common/compat.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_median_ops(py::module_& m);
