#pragma once

#include "common/compat.h"

#include <pybind11/pybind11.h>

void bind_median_reduce_metal_ops(pybind11::module_& m);
