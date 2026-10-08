#pragma once

#include "common/compat.h"
#include "common/default_init_vector.h"

#include <pybind11/pybind11.h>

#include <cstdint>
#include <vector>

namespace py = pybind11;

namespace hnw::wavelet {

using Buffer = DefaultInitVector<double>;

struct CpuImage {
    Buffer values;
    int64_t height = 0;
    int64_t width = 0;
};

CpuImage dec_rec_cpu(const double* image, int64_t height, int64_t width, int64_t level);

} // namespace hnw::wavelet

void bind_wavelet_ops(py::module_& m);
