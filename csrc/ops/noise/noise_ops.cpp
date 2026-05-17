#include "noise_ops.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>

namespace {

#ifndef HNW_ENABLE_OMP_SIMD
#define HNW_ENABLE_OMP_SIMD 0
#endif

#if defined(_MSC_VER)
#define HNW_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define HNW_RESTRICT __restrict__
#else
#define HNW_RESTRICT
#endif

void validate_same_shape(const py::array& first,
                         const py::array& second,
                         const char* op_name) {
    if (first.ndim() != second.ndim()) {
        throw std::invalid_argument(std::string(op_name) + ": ndim mismatch");
    }
    for (ssize_t i = 0; i < first.ndim(); ++i) {
        if (first.shape(i) != second.shape(i)) {
            throw std::invalid_argument(std::string(op_name) + ": shape mismatch");
        }
    }
}

template <typename T>
void equalize_noise_correct_kernel(
    py::buffer_info& out_info,
    const py::buffer_info& max_info,
    const py::buffer_info& filled_std_info,
    const double sigma_ref,
    const double c_n_eff,
    const double max_value,
    const double highlight_preserve) {
    auto* HNW_RESTRICT out_ptr = static_cast<T*>(out_info.ptr);
    const auto* HNW_RESTRICT max_ptr = static_cast<const T*>(max_info.ptr);
    const auto* HNW_RESTRICT filled_std_ptr =
        static_cast<const T*>(filled_std_info.ptr);
    const ssize_t total = out_info.size;
    const T sigma_ref_value = static_cast<T>(sigma_ref);
    const T c_n_eff_value = static_cast<T>(c_n_eff);
    const T max_value_value = static_cast<T>(max_value);
    const T highlight_value = static_cast<T>(highlight_preserve);
    const T denom = max_value_value * (static_cast<T>(1) - highlight_value);

    py::gil_scoped_release release;
#if defined(_OPENMP) && HNW_ENABLE_OMP_SIMD
#pragma omp parallel for simd schedule(static)
#elif defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ssize_t i = 0; i < total; ++i) {
        const T max_pixel = max_ptr[i];
        const T numerator = std::min(
            static_cast<T>(0),
            max_value_value * highlight_value - max_pixel);
        const T fix_strength = numerator / denom + static_cast<T>(1);
        const T fixed_std = fix_strength * filled_std_ptr[i];
        const T corrected =
            max_pixel - (fixed_std - sigma_ref_value) * c_n_eff_value;
        out_ptr[i] = std::clamp(corrected, static_cast<T>(0), max_value_value);
    }
}

template <typename T>
py::array_t<T> equalize_noise_correct_impl(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& max_img,
    const py::array_t<T, py::array::c_style | py::array::forcecast>&
        filled_std_img,
    const double sigma_ref,
    const double c_n_eff,
    const double max_value,
    const double highlight_preserve) {
    validate_same_shape(max_img, filled_std_img, "equalize_noise_correct");
    if (!(highlight_preserve >= 0.0 && highlight_preserve < 1.0)) {
        throw std::invalid_argument(
            "equalize_noise_correct: highlight_preserve must be in [0, 1)");
    }

    auto max_info = max_img.request();
    auto filled_std_info = filled_std_img.request();
    py::array_t<T> out(max_info.shape);
    auto out_info = out.request();
    equalize_noise_correct_kernel<T>(out_info,
                                     max_info,
                                     filled_std_info,
                                     sigma_ref,
                                     c_n_eff,
                                     max_value,
                                     highlight_preserve);
    return out;
}

py::array equalize_noise_correct_dispatch(
    const py::array& max_img,
    const py::array& filled_std_img,
    const double sigma_ref,
    const double c_n_eff,
    const double max_value,
    const double highlight_preserve) {
    const std::string max_dtype = py::str(max_img.dtype()).cast<std::string>();
    if (max_dtype != py::str(filled_std_img.dtype()).cast<std::string>()) {
        throw std::invalid_argument("equalize_noise_correct: dtype mismatch");
    }

    if (py::isinstance<py::array_t<float>>(max_img)) {
        return equalize_noise_correct_impl<float>(
            max_img.cast<py::array_t<float>>(),
            filled_std_img.cast<py::array_t<float>>(),
            sigma_ref,
            c_n_eff,
            max_value,
            highlight_preserve);
    }
    if (py::isinstance<py::array_t<double>>(max_img)) {
        return equalize_noise_correct_impl<double>(
            max_img.cast<py::array_t<double>>(),
            filled_std_img.cast<py::array_t<double>>(),
            sigma_ref,
            c_n_eff,
            max_value,
            highlight_preserve);
    }

    throw std::invalid_argument(
        "equalize_noise_correct: unsupported dtype");
}

}  // namespace

void bind_noise_ops(py::module_& m) {
    m.def("equalize_noise_correct",
          &equalize_noise_correct_dispatch,
          py::arg("max_img"),
          py::arg("filled_std_img"),
          py::arg("sigma_ref"),
          py::arg("c_n_eff"),
          py::arg("max_value"),
          py::arg("highlight_preserve"),
          "Apply highlight-preserving equalize-noise correction with a C++ pixel kernel.");
}
