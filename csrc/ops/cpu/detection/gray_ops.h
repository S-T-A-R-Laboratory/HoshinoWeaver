#pragma once

#include "common/compat.h"

#include <pybind11/pybind11.h>

void bind_detection_gray_cpu_ops(pybind11::module_& m);
