#include "gray_ops.h"

#include "common/compat.h"

#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

// These streaming casts saturate host bandwidth with a small team.
int copy_threads() {
#if defined(_OPENMP)
    return std::min(omp_get_max_threads(), 8);
#else
    return 1;
#endif
}

template <typename Input, typename Output>
py::array_t<Output> cast_image(const py::array& image, const double scale) {
    const std::vector<ssize_t> shape(image.shape(), image.shape() + image.ndim());
    py::array_t<Output> out(shape);
    const auto* source = static_cast<const Input*>(image.data());
    Output* target = out.mutable_data();
    const ssize_t count = image.size();
    const int threads = copy_threads();
    (void)threads;
    {
        py::gil_scoped_release release;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(threads) if (count >= 65536)
#endif
        for (ssize_t i = 0; i < count; ++i) {
            if constexpr (std::is_same_v<Output, float>) {
                target[i] = static_cast<float>(source[i]);
            } else {
                target[i] = static_cast<double>(source[i]) / scale;
            }
        }
    }
    return out;
}

py::object quantize_gray(const py::array& image) {
    const std::vector<ssize_t> shape(image.shape(), image.shape() + image.ndim());
    py::array_t<uint16_t> out(shape);
    const auto* source = static_cast<const float*>(image.data());
    auto* target = out.mutable_data();
    const ssize_t count = image.size();
    const int threads = copy_threads();
    (void)threads;
    int invalid = 0;
    {
        py::gil_scoped_release release;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(threads) if (count >= 65536)                 \
    reduction(| : invalid)
#endif
        for (ssize_t i = 0; i < count; ++i) {
            const float value = source[i];
            if (!(value >= 0.0f && value <= 65535.0f)) {
                invalid = 1;
                continue;
            }
            // Nearest-even, including half-integers, without relying on fenv.
            uint32_t rounded = static_cast<uint32_t>(value);
            const float fraction = value - static_cast<float>(rounded);
            if (fraction > 0.5f || (fraction == 0.5f && (rounded & 1U) != 0)) {
                ++rounded;
            }
            target[i] = static_cast<uint16_t>(rounded);
        }
    }
    if (invalid != 0) {
        return py::none();
    }
    return out;
}

py::object detection_gray_cast_cpu(const py::array& image, const std::string& stage,
                                   const double scale) {
    if (!(image.flags() & py::array::c_style)) {
        throw std::invalid_argument("detection_gray: input must be C-contiguous");
    }
    if (image.ndim() != 2 && image.ndim() != 3) {
        throw std::invalid_argument("detection_gray: input must have 2 or 3 dimensions");
    }
    if (stage == "prepare") {
        if (image.dtype().is(py::dtype::of<uint8_t>())) {
            return cast_image<uint8_t, float>(image, 1.0);
        }
        if (image.dtype().is(py::dtype::of<uint16_t>())) {
            return cast_image<uint16_t, float>(image, 1.0);
        }
    } else if (stage == "normalize") {
        if (!(scale > 0.0) || !std::isfinite(scale)) {
            throw std::invalid_argument("detection_gray: scale must be positive and finite");
        }
        if (image.dtype().is(py::dtype::of<uint8_t>())) {
            return cast_image<uint8_t, double>(image, scale);
        }
        if (image.dtype().is(py::dtype::of<uint16_t>())) {
            return cast_image<uint16_t, double>(image, scale);
        }
        if (image.dtype().is(py::dtype::of<float>())) {
            return cast_image<float, double>(image, scale);
        }
    } else if (stage == "quantize" && image.dtype().is(py::dtype::of<float>())) {
        return quantize_gray(image);
    }
    throw std::invalid_argument("detection_gray: unsupported stage or input dtype");
}

} // namespace

void bind_detection_gray_cpu_ops(py::module_& m) {
    m.def("detection_gray_cast_cpu", &detection_gray_cast_cpu, py::arg("image"), py::arg("stage"),
          py::arg("scale") = 1.0,
          "Parallel casts around OpenCV grayscale conversion; quantize declines invalid ranges.");
}
