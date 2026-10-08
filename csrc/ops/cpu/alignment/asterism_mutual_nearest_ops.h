#pragma once

#include "common/compat.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <vector>

namespace py = pybind11;

namespace hnw::asterism {

using TokenArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

// Token points bucketed into a dense grid of cubic cells whose edge slightly
// exceeds the match threshold, so every point within the threshold of a query
// lies in the 3x3x3 cells around the query's cell. Both token sets share one
// frame (origin, cell, dims). Points of a cell are stored contiguously (CSR).
struct GridFrame {
    double origin[3] = {0.0, 0.0, 0.0};
    double cell = 0.0;
    int64_t dims[3] = {1, 1, 1};
};

struct TokenGrid {
    std::vector<int32_t> cell_start; // cells + 1 offsets into points
    std::vector<int32_t> points;
    std::vector<double> coords; // point coordinates in slot order (3 per slot)
};

constexpr int64_t kMaxGridCells = int64_t{1} << 22;

// Bounded-nearest result when several points share the nearest distance:
// SciPy's choice among them is unspecified, so callers use the tree path.
constexpr int32_t kAmbiguousNearest = -2;

// Throws on malformed shapes or a non-positive threshold.
void validate_inputs(const TokenArray& values1, const TokenArray& values2, double threshold);

// Frame from the finite per-axis extent of both sets. Returns false when the
// dense grid would exceed kMaxGridCells; callers then use the tree-based
// reference path.
bool finish_token_grid_frame(const double lo[3], const double hi[3], double threshold,
                             GridFrame* frame);

// Also returns false when any value is non-finite.
bool token_grid_frame(const double* values1, int64_t n1, const double* values2, int64_t n2,
                      double threshold, GridFrame* frame);

TokenGrid build_token_grid(const double* values, int64_t n, const GridFrame& frame);

int64_t token_cell_coordinate(double value, double origin, double cell);

// Squared distance accumulated in SciPy cKDTree's order: ((0 + d0^2) + d1^2) + d2^2.
inline double token_squared_distance(const double* a, const double* b) {
    const double d0 = a[0] - b[0];
    const double d1 = a[1] - b[1];
    const double d2 = a[2] - b[2];
    double sum = 0.0;
    sum += d0 * d0;
    sum += d1 * d1;
    sum += d2 * d2;
    return sum;
}

// (pairs1, pairs2) int64 arrays in ascending pairs1 from mutual12[i] = j for
// mutual pairs and -1 otherwise. Requires the GIL.
py::tuple mutual_pairs_to_arrays(const std::vector<int32_t>& mutual12);

py::tuple empty_pairs();

bool has_ambiguous_nearest(const std::vector<int32_t>& nearest);

} // namespace hnw::asterism

void bind_asterism_mutual_nearest_cpu_ops(py::module_& m);
