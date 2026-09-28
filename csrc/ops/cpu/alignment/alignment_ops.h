#pragma once

#include "common/compat.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace hnw::alignment {

using PointArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

constexpr ssize_t kFeatureBins = 120;

// Validates extract_point_features inputs; returns the neighbour pool size
// min(2k, N), or 0 when there are no points.
ssize_t point_feature_pool_size(const PointArray& vec, const PointArray& vol, int k);

} // namespace hnw::alignment

void bind_alignment_ops(py::module_& m);
