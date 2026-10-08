#pragma once

#include "ops/cpu/alignment/asterism_mutual_nearest_ops.h"

#include <pybind11/pybind11.h>

#include <cstdint>
#include <vector>

namespace py = pybind11;

// Builds both token grids on the device and fills mutual12[i] with the mutual
// nearest j of token i, or -1. Returns false when the token extent cannot be
// gridded (non-finite values or more than kMaxGridCells cells) or a nearest
// distance within the threshold ties.
bool launch_asterism_mutual_nearest_cuda(const double* values1, int64_t n1, const double* values2,
                                         int64_t n2, double threshold,
                                         std::vector<int32_t>* mutual12);

void bind_asterism_mutual_nearest_cuda_ops(py::module_& m);
