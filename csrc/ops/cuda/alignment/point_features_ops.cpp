#include "point_features_ops.h"

#include "common/compat.h"
#include "ops/cpu/alignment/alignment_ops.h"

#include <pybind11/numpy.h>

#include <vector>

namespace {

using hnw::alignment::kFeatureBins;
using hnw::alignment::PointArray;

py::object extract_point_features_cuda_impl(const PointArray& vec, const PointArray& vol,
                                            const int k) {
    const ssize_t pool = hnw::alignment::point_feature_pool_size(vec, vol, k);
    if (pool == 0) {
        return py::array_t<double>(std::vector<ssize_t>{0, kFeatureBins});
    }
    if (pool > kPointFeatureMaxPool) {
        return py::none();
    }
    const ssize_t n = vec.shape(0);
    py::array_t<double> out(std::vector<ssize_t>{n, kFeatureBins});
    const double* vec_ptr = vec.data();
    const double* vol_ptr = vol.data();
    double* out_ptr = out.mutable_data();
    bool exact = true;
    {
        py::gil_scoped_release release;
        const hnw::alignment::DirectionGrid grid(vec_ptr, n, pool);
        exact = launch_extract_point_features_cuda(vec_ptr, vol_ptr, n, k, static_cast<int>(pool),
                                                   grid, out_ptr);
    }
    if (!exact) {
        return py::none();
    }
    return out;
}

} // namespace

void bind_point_features_cuda_ops(py::module_& m) {
    m.def("extract_point_features_cuda", &extract_point_features_cuda_impl, py::arg("vec"),
          py::arg("vol"), py::arg("k") = 15,
          "Extract star-point geometric descriptors using CUDA; returns None when 2k "
          "exceeds the device neighbour pool or a near-tied neighbour ordering needs the CPU.");
}
