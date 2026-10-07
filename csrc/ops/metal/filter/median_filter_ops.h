#pragma once

#include "common/compat.h"

#include <pybind11/pybind11.h>

void bind_median_filter_metal_ops(pybind11::module_& m);
