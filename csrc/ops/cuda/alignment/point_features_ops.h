#pragma once

#include "ops/cpu/alignment/direction_grid.h"

#include <pybind11/pybind11.h>

#include <cstdint>

namespace py = pybind11;

// Largest neighbour pool (2k) the CUDA selection keeps per point.
constexpr int kPointFeatureMaxPool = 32;

// Writes the (n, 120) descriptors of extract_point_features into out (host).
// `grid` supplies the candidate cells; an unusable grid means a full scan.
// Returns false, leaving out unspecified, when a near tie in the vol*rho
// ordering could select differently from the CPU backend.
bool launch_extract_point_features_cuda(const double* vec, const double* vol, int64_t n, int k,
                                        int pool, const hnw::alignment::DirectionGrid& grid,
                                        double* out);

void bind_point_features_cuda_ops(py::module_& m);
