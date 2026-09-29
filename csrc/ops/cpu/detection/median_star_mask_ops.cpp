#include "median_star_mask_ops.h"

#include "common/cpu_compat.h"
#include "ops/cpu/filter/median_filter_internal.h"

#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

constexpr ssize_t MAX_MEDIAN_FILTER_KSIZE = 65535;

void validate_odd_ksize(ssize_t ksize, const char* name, bool allow_zero) {
    if (allow_zero && ksize == 0) {
        return;
    }
    if (ksize <= 0 || (ksize % 2) == 0) {
        const char* requirement = allow_zero ? " must be zero or a positive odd integer"
                                             : " must be a positive odd integer";
        throw std::invalid_argument(std::string("median_star_mask: ") + name + requirement);
    }
    if (ksize > MAX_MEDIAN_FILTER_KSIZE) {
        throw std::invalid_argument(std::string("median_star_mask: ") + name + " is too large");
    }
}

template <typename T> void quantize_gray(const T* input, uint16_t* output, ssize_t total) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ssize_t index = 0; index < total; ++index) {
        if constexpr (std::is_same_v<T, uint8_t>) {
            output[index] = static_cast<uint16_t>(input[index]) * 257U;
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            output[index] = input[index];
        } else if constexpr (std::is_same_v<T, float>) {
            output[index] = static_cast<uint16_t>(std::nearbyint(input[index] * 65535.0F));
        } else {
            const float value = static_cast<float>(input[index]);
            output[index] = static_cast<uint16_t>(std::nearbyint(value * 65535.0F));
        }
    }
}

template <typename T> bool normalized_float_input_is_valid(const T* input, ssize_t total) {
    int invalid = 0;
#if defined(_OPENMP)
#pragma omp parallel for reduction(| : invalid) schedule(static)
#endif
    for (ssize_t index = 0; index < total; ++index) {
        const double value = static_cast<double>(input[index]);
        invalid |= !std::isfinite(value) || value < 0.0 || value > 1.0;
    }
    return invalid == 0;
}

bool cross_value(const uint8_t* input, ssize_t height, ssize_t width, ssize_t y, ssize_t x,
                 ssize_t radius, bool erode) {
    bool value = erode;
    for (ssize_t delta = -radius; delta <= radius; ++delta) {
        const ssize_t xx = x + delta;
        if (xx >= 0 && xx < width) {
            const bool set = input[y * width + xx] != 0;
            value = erode ? value && set : value || set;
        }
        const ssize_t yy = y + delta;
        if (delta != 0 && yy >= 0 && yy < height) {
            const bool set = input[yy * width + x] != 0;
            value = erode ? value && set : value || set;
        }
    }
    return value;
}

void cross_morphology(const uint8_t* input, uint8_t* output, ssize_t height, ssize_t width,
                      ssize_t ksize, bool erode) {
    const ssize_t radius = ksize / 2;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ssize_t y = 0; y < height; ++y) {
        for (ssize_t x = 0; x < width; ++x) {
            output[y * width + x] = cross_value(input, height, width, y, x, radius, erode) ? 1 : 0;
        }
    }
}

template <typename T>
py::tuple
median_star_mask_impl(const py::array_t<T, py::array::c_style | py::array::forcecast>& image,
                      ssize_t median_ksize, double threshold_ratio, ssize_t open_ksize,
                      ssize_t dilate_ksize, const py::object& mask_object,
                      const py::object& background_object) {
    if (image.ndim() != 2 || image.shape(0) <= 0 || image.shape(1) <= 0) {
        throw std::invalid_argument("median_star_mask: image must be a non-empty 2D array");
    }
    validate_odd_ksize(median_ksize, "median_ksize", false);
    validate_odd_ksize(open_ksize, "open_ksize", true);
    validate_odd_ksize(dilate_ksize, "dilate_ksize", true);
    if (!std::isfinite(threshold_ratio)) {
        throw std::invalid_argument("median_star_mask: threshold_ratio must be finite");
    }

    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask;
    const uint8_t* mask_ptr = nullptr;
    if (!mask_object.is_none()) {
        mask = mask_object.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
        if (mask.ndim() != 2 || mask.shape(0) != image.shape(0) ||
            mask.shape(1) != image.shape(1)) {
            throw std::invalid_argument("median_star_mask: mask shape must match image");
        }
        mask_ptr = mask.data();
    }

    const ssize_t height = image.shape(0);
    const ssize_t width = image.shape(1);
    if (height > std::numeric_limits<ssize_t>::max() / width) {
        throw std::invalid_argument("median_star_mask: image is too large");
    }
    const ssize_t total = height * width;
    std::vector<uint16_t> working;
    std::vector<uint16_t> background;
    py::array_t<uint16_t, py::array::c_style | py::array::forcecast> supplied_background;
    const bool has_precomputed_background = !background_object.is_none();
    const uint16_t* working_ptr = nullptr;
    const uint16_t* background_ptr = nullptr;
    if (!has_precomputed_background) {
        working.resize(static_cast<size_t>(total));
        background.resize(static_cast<size_t>(total));
        working_ptr = working.data();
        background_ptr = background.data();
    } else {
        const py::array candidate = background_object.cast<py::array>();
        if (!candidate.dtype().is(py::dtype::of<uint16_t>()) || candidate.ndim() != 2 ||
            candidate.shape(0) != height || candidate.shape(1) != width) {
            throw std::invalid_argument("median_star_mask: background must be uint16 image shape");
        }
        supplied_background =
            candidate.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>();
        background_ptr = supplied_background.data();
        if constexpr (std::is_same_v<T, uint16_t>) {
            working_ptr = image.data();
        } else {
            throw std::invalid_argument(
                "median_star_mask: precomputed background requires uint16 input");
        }
    }
    py::array_t<float> response({height, width});
    py::array_t<uint8_t> star_mask({height, width});
    auto response_info = response.request();
    auto star_mask_info = star_mask.request();
    auto* response_ptr = static_cast<float*>(response_info.ptr);
    auto* star_mask_ptr = static_cast<uint8_t*>(star_mask_info.ptr);
    const T* input_ptr = image.data();
    double threshold = 0.0;

    {
        py::gil_scoped_release release;
        if constexpr (std::is_floating_point_v<T>) {
            if (!normalized_float_input_is_valid(input_ptr, total)) {
                throw std::invalid_argument("median_star_mask: floating-point image must be finite "
                                            "and normalized to [0, 1]");
            }
        }
        if (!has_precomputed_background) {
            quantize_gray(input_ptr, working.data(), total);
            hnw::cpu::median_filter_2d_kernel<uint16_t>(working.data(), background.data(), height,
                                                        width, 1, median_ksize);
        }

        double sum = 0.0;
        uint64_t valid_count = 0;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : sum, valid_count) schedule(static)
#endif
        for (ssize_t index = 0; index < total; ++index) {
            const float value = static_cast<float>(
                (static_cast<int32_t>(working_ptr[static_cast<size_t>(index)]) -
                 static_cast<int32_t>(background_ptr[static_cast<size_t>(index)])) /
                65535.0);
            response_ptr[index] = value;
            if (mask_ptr == nullptr || mask_ptr[index] != 0) {
                sum += static_cast<double>(value);
                ++valid_count;
            }
        }
        if (valid_count == 0) {
            throw std::invalid_argument("median_star_mask: mask selects no pixels");
        }
        // The compiled path intentionally accumulates in double. NumPy's
        // float32 std may differ by a few ulps, but mask decisions are locked
        // against the reference path by focused and real-image quality gates.
        const double mean = sum / static_cast<double>(valid_count);
        double squared_sum = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : squared_sum) schedule(static)
#endif
        for (ssize_t index = 0; index < total; ++index) {
            if (mask_ptr == nullptr || mask_ptr[index] != 0) {
                const double delta = static_cast<double>(response_ptr[index]) - mean;
                squared_sum += delta * delta;
            }
        }
        threshold = std::sqrt(squared_sum / static_cast<double>(valid_count)) * threshold_ratio;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (ssize_t index = 0; index < total; ++index) {
            star_mask_ptr[index] =
                (response_ptr[index] > threshold && (mask_ptr == nullptr || mask_ptr[index] != 0))
                    ? 1
                    : 0;
        }

        std::vector<uint8_t> scratch;
        if (open_ksize > 0 || dilate_ksize > 0) {
            scratch.resize(static_cast<size_t>(total));
        }
        if (open_ksize > 0) {
            cross_morphology(star_mask_ptr, scratch.data(), height, width, open_ksize, true);
            cross_morphology(scratch.data(), star_mask_ptr, height, width, open_ksize, false);
        }
        if (dilate_ksize > 0) {
            cross_morphology(star_mask_ptr, scratch.data(), height, width, dilate_ksize, false);
            std::copy(scratch.begin(), scratch.end(), star_mask_ptr);
        }
    }

    return py::make_tuple(std::move(star_mask), std::move(response), threshold);
}

py::tuple median_star_mask_dispatch(const py::array& image, ssize_t median_ksize,
                                    double threshold_ratio, ssize_t open_ksize,
                                    ssize_t dilate_ksize, const py::object& mask) {
    if (py::isinstance<py::array_t<uint8_t>>(image)) {
        return median_star_mask_impl<uint8_t>(image.cast<py::array_t<uint8_t>>(), median_ksize,
                                              threshold_ratio, open_ksize, dilate_ksize, mask,
                                              py::none());
    }
    if (py::isinstance<py::array_t<uint16_t>>(image)) {
        return median_star_mask_impl<uint16_t>(image.cast<py::array_t<uint16_t>>(), median_ksize,
                                               threshold_ratio, open_ksize, dilate_ksize, mask,
                                               py::none());
    }
    if (py::isinstance<py::array_t<float>>(image)) {
        return median_star_mask_impl<float>(image.cast<py::array_t<float>>(), median_ksize,
                                            threshold_ratio, open_ksize, dilate_ksize, mask,
                                            py::none());
    }
    if (py::isinstance<py::array_t<double>>(image)) {
        return median_star_mask_impl<double>(image.cast<py::array_t<double>>(), median_ksize,
                                             threshold_ratio, open_ksize, dilate_ksize, mask,
                                             py::none());
    }
    throw std::invalid_argument(
        "median_star_mask: unsupported dtype; expected uint8/uint16/float32/float64");
}

py::tuple
median_star_mask_with_background_dispatch(const py::array& image, const py::array& background,
                                          const ssize_t median_ksize, const double threshold_ratio,
                                          const ssize_t open_ksize, const ssize_t dilate_ksize,
                                          const py::object& mask) {
    if (!image.dtype().is(py::dtype::of<uint16_t>())) {
        throw std::invalid_argument(
            "median_star_mask: precomputed background requires uint16 input");
    }
    return median_star_mask_impl<uint16_t>(image.cast<py::array_t<uint16_t>>(), median_ksize,
                                           threshold_ratio, open_ksize, dilate_ksize, mask,
                                           background);
}

} // namespace

void bind_median_star_mask_cpu_ops(py::module_& m) {
    m.def("median_star_mask_cpu", &median_star_mask_dispatch, py::arg("image"),
          py::arg("median_ksize"), py::arg("threshold_ratio"), py::arg("open_ksize") = 3,
          py::arg("dilate_ksize") = 0, py::arg("mask") = py::none(),
          "Build a median-background star mask and signed response on CPU.");
    m.def("median_star_mask_with_background_cpu", &median_star_mask_with_background_dispatch,
          py::arg("image"), py::arg("background"), py::arg("median_ksize"),
          py::arg("threshold_ratio"), py::arg("open_ksize") = 3, py::arg("dilate_ksize") = 0,
          py::arg("mask") = py::none(),
          "Build the star mask and response from an exact precomputed uint16 median background.");
}
