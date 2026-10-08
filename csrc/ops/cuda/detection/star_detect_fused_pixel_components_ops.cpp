#include "star_detect_fused_pixel_components_ops.h"

#include "common/compat.h"
#include "common/wavelet_geometry.h"

#include <pybind11/numpy.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using GrayArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

// A uint8/uint16 image of shape (H, W) or (H, W, 3) BGR, kept contiguous.
struct HostSource {
    py::array array;
    StarDetectSource source;
};

HostSource convertible_source(const py::array& image) {
    HostSource out;
    if (image.dtype().is(py::dtype::of<uint8_t>())) {
        out.array = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>::ensure(image);
        out.source.sample_bytes = 1;
    } else if (image.dtype().is(py::dtype::of<uint16_t>())) {
        out.array = py::array_t<uint16_t, py::array::c_style | py::array::forcecast>::ensure(image);
        out.source.sample_bytes = 2;
    } else {
        throw std::invalid_argument(
            "star_detect_fused_pixel_components: source must be uint8 or uint16");
    }
    if (!(out.array.ndim() == 2 || (out.array.ndim() == 3 && out.array.shape(2) == 3))) {
        throw std::invalid_argument(
            "star_detect_fused_pixel_components: source must have shape (H, W) or (H, W, 3)");
    }
    out.source.channels = out.array.ndim() == 3 ? 3 : 1;
    out.source.data = out.array.data();
    return out;
}

py::object detect_impl(const StarDetectSource& source, const ssize_t height, const ssize_t width,
                       py::object mask_obj, const ssize_t small_height, const ssize_t small_width,
                       const ssize_t level, const GrayArray& gaussian_kernel) {
    if (height <= 0 || width <= 0 || small_height <= 0 || small_width <= 0) {
        throw std::invalid_argument("star_detect_fused_pixel_components: image "
                                    "dimensions must be positive");
    }
    if (height > std::numeric_limits<int>::max() || width > std::numeric_limits<int>::max() ||
        small_height > std::numeric_limits<int>::max() ||
        small_width > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("star_detect_fused_pixel_components: image shape is too large");
    }
    if (height > std::numeric_limits<int>::max() / width ||
        small_height > std::numeric_limits<int>::max() / small_width) {
        throw std::invalid_argument("star_detect_fused_pixel_components: image is too large");
    }
    if (level <= 0 || level > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("star_detect_fused_pixel_components: invalid wavelet level");
    }
    if (gaussian_kernel.ndim() != 1 || gaussian_kernel.shape(0) <= 0 ||
        gaussian_kernel.shape(0) > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            "star_detect_fused_pixel_components: gaussian kernel must be 1D");
    }
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask;
    const uint8_t* mask_ptr = nullptr;
    if (!mask_obj.is_none()) {
        mask = mask_obj.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
        if (mask.ndim() != 2 || mask.shape(0) != height || mask.shape(1) != width) {
            throw std::invalid_argument(
                "star_detect_fused_pixel_components: mask shape must match image");
        }
        mask_ptr = mask.data();
    }

    const auto [small_out_h, small_out_w] =
        hnw::wavelet::reconstructed_shape(small_height, small_width, level);
    const ssize_t small_output_size = small_out_h * small_out_w;
    if (small_out_h <= 0 || small_out_w <= 0 || small_out_h > std::numeric_limits<int>::max() ||
        small_out_w > std::numeric_limits<int>::max() ||
        small_output_size > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            "star_detect_fused_pixel_components: wavelet output shape is invalid");
    }

    std::vector<double> positions_xy;
    std::vector<double> intensities;
    py::array_t<uint8_t> binary_mask(std::vector<ssize_t>{height, width});
    bool detected = true;
    {
        py::gil_scoped_release release;
        detected = launch_star_detect_fused_pixel_components(
            source, mask_ptr, gaussian_kernel.data(), &positions_xy, &intensities,
            binary_mask.mutable_data(), static_cast<int>(height), static_cast<int>(width),
            static_cast<int>(small_height), static_cast<int>(small_width), static_cast<int>(level),
            static_cast<int>(gaussian_kernel.shape(0)));
    }
    if (!detected) {
        return py::none();
    }

    const ssize_t out_count = static_cast<ssize_t>(intensities.size());
    py::array_t<double> positions({out_count, static_cast<ssize_t>(2)});
    py::array_t<double> intensities_out({out_count});
    std::copy(positions_xy.begin(), positions_xy.end(), positions.mutable_data());
    std::copy(intensities.begin(), intensities.end(), intensities_out.mutable_data());
    return py::make_tuple(positions, intensities_out, binary_mask);
}

py::object star_detect_fused_pixel_components_impl(const GrayArray& image, py::object mask_obj,
                                                   const ssize_t small_height,
                                                   const ssize_t small_width, const ssize_t level,
                                                   const GrayArray& gaussian_kernel) {
    if (image.ndim() != 2) {
        throw std::invalid_argument("star_detect_fused_pixel_components: image must be 2D");
    }
    StarDetectSource source;
    source.data = image.data();
    return detect_impl(source, image.shape(0), image.shape(1), std::move(mask_obj), small_height,
                       small_width, level, gaussian_kernel);
}

py::object star_detect_fused_pixel_components_source_impl(
    const py::array& image, py::object mask_obj, const ssize_t small_height,
    const ssize_t small_width, const ssize_t level, const GrayArray& gaussian_kernel) {
    const HostSource host = convertible_source(image);
    return detect_impl(host.source, host.array.shape(0), host.array.shape(1), std::move(mask_obj),
                       small_height, small_width, level, gaussian_kernel);
}

py::array_t<double> star_detect_source_gray_impl(const py::array& image) {
    const HostSource host = convertible_source(image);
    const ssize_t height = host.array.shape(0);
    const ssize_t width = host.array.shape(1);
    if (height <= 0 || width <= 0 ||
        height > std::numeric_limits<int>::max() / std::max<ssize_t>(width, 1)) {
        throw std::invalid_argument("star_detect_source_gray: invalid image shape");
    }
    py::array_t<double> gray(std::vector<ssize_t>{height, width});
    {
        py::gil_scoped_release release;
        launch_star_detect_source_gray(host.source, static_cast<int>(height),
                                       static_cast<int>(width), gray.mutable_data());
    }
    return gray;
}

} // namespace

void bind_star_detect_fused_pixel_components_ops(py::module_& m) {
    m.def("star_detect_fused_pixel_components_cuda", &star_detect_fused_pixel_components_impl,
          py::arg("image"), py::arg("mask"), py::arg("small_height"), py::arg("small_width"),
          py::arg("level"), py::arg("gaussian_kernel"));
    m.def("star_detect_fused_pixel_components_cuda_from_source",
          &star_detect_fused_pixel_components_source_impl, py::arg("source"), py::arg("mask"),
          py::arg("small_height"), py::arg("small_width"), py::arg("level"),
          py::arg("gaussian_kernel"),
          "Fused detection of a uint8/uint16 (H, W) or (H, W, 3) BGR image whose gray is "
          "converted on the device like to_gray_f64; returns None when that gray is constant.");
    m.def("star_detect_source_gray_cuda", &star_detect_source_gray_impl, py::arg("source"),
          "Device to_gray_f64 of a uint8/uint16 image, for verifying the conversion.");
}
