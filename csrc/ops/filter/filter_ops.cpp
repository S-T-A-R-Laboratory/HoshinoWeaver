#include "filter_ops.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>

namespace {

constexpr ssize_t MAX_MEDIAN_FILTER_KSIZE = 65535;

#if defined(_MSC_VER)
#define HNW_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define HNW_RESTRICT __restrict__
#else
#define HNW_RESTRICT
#endif

inline ssize_t clamp_index(ssize_t value, ssize_t low, ssize_t high) {
    return std::max(low, std::min(value, high));
}

template <typename T>
struct MedianHistogram;

template <>
struct MedianHistogram<uint8_t> {
    std::array<uint32_t, 256> bins{};

    void clear() {
        bins.fill(0);
    }

    void add(uint8_t value) {
        ++bins[value];
    }

    void remove(uint8_t value) {
        --bins[value];
    }

    uint8_t median(uint32_t target_rank) const {
        uint32_t count = 0;
        for (uint32_t value = 0; value < bins.size(); ++value) {
            count += bins[value];
            if (count >= target_rank) {
                return static_cast<uint8_t>(value);
            }
        }
        return 255;
    }
};

template <>
struct MedianHistogram<uint16_t> {
    std::array<uint32_t, 256> coarse{};
    std::vector<uint32_t> fine;

    MedianHistogram() : fine(65536, 0) {}

    void clear() {
        coarse.fill(0);
        std::fill(fine.begin(), fine.end(), 0);
    }

    void add(uint16_t value) {
        ++coarse[value >> 8];
        ++fine[value];
    }

    void remove(uint16_t value) {
        --coarse[value >> 8];
        --fine[value];
    }

    uint16_t median(uint32_t target_rank) const {
        uint32_t count = 0;
        uint32_t high_bin = 0;
        for (; high_bin < coarse.size(); ++high_bin) {
            const uint32_t next = count + coarse[high_bin];
            if (next >= target_rank) {
                break;
            }
            count = next;
        }

        const uint32_t start = high_bin << 8;
        const uint32_t end = start + 256;
        for (uint32_t value = start; value < end; ++value) {
            count += fine[value];
            if (count >= target_rank) {
                return static_cast<uint16_t>(value);
            }
        }
        return 65535;
    }
};

template <typename T>
inline T load_pixel(
    const T* HNW_RESTRICT input,
    ssize_t h,
    ssize_t w,
    ssize_t channels,
    ssize_t y,
    ssize_t x,
    ssize_t c) {
    const ssize_t yy = clamp_index(y, 0, h - 1);
    const ssize_t xx = clamp_index(x, 0, w - 1);
    return input[(yy * w + xx) * channels + c];
}

template <typename T>
void build_histogram_for_row(
    MedianHistogram<T>& hist,
    const T* HNW_RESTRICT input,
    ssize_t h,
    ssize_t w,
    ssize_t channels,
    ssize_t y,
    ssize_t c,
    ssize_t radius) {
    hist.clear();
    for (ssize_t dy = -radius; dy <= radius; ++dy) {
        for (ssize_t dx = -radius; dx <= radius; ++dx) {
            hist.add(load_pixel(input, h, w, channels, y + dy, dx, c));
        }
    }
}

template <typename T>
void slide_histogram_right(
    MedianHistogram<T>& hist,
    const T* HNW_RESTRICT input,
    ssize_t h,
    ssize_t w,
    ssize_t channels,
    ssize_t y,
    ssize_t x,
    ssize_t c,
    ssize_t radius) {
    const ssize_t remove_x = x - radius;
    const ssize_t add_x = x + radius + 1;
    for (ssize_t dy = -radius; dy <= radius; ++dy) {
        hist.remove(load_pixel(input, h, w, channels, y + dy, remove_x, c));
        hist.add(load_pixel(input, h, w, channels, y + dy, add_x, c));
    }
}

template <typename T>
void median_filter_2d_kernel(
    const T* HNW_RESTRICT input,
    T* HNW_RESTRICT output,
    ssize_t h,
    ssize_t w,
    ssize_t channels,
    ssize_t ksize) {
    const ssize_t radius = ksize / 2;
    const uint64_t window_area =
        static_cast<uint64_t>(ksize) * static_cast<uint64_t>(ksize);
    const uint32_t target_rank = static_cast<uint32_t>(window_area / 2 + 1);
    const ssize_t task_count = h * channels;

#if defined(_OPENMP)
#pragma omp parallel
    {
        MedianHistogram<T> hist;
#pragma omp for schedule(static)
        for (ssize_t task = 0; task < task_count; ++task) {
            const ssize_t y = task / channels;
            const ssize_t c = task % channels;
            build_histogram_for_row(hist, input, h, w, channels, y, c, radius);
            for (ssize_t x = 0; x < w; ++x) {
                output[(y * w + x) * channels + c] = hist.median(target_rank);
                if (x + 1 < w) {
                    slide_histogram_right(
                        hist, input, h, w, channels, y, x, c, radius);
                }
            }
        }
    }
#else
    MedianHistogram<T> hist;
    for (ssize_t task = 0; task < task_count; ++task) {
        const ssize_t y = task / channels;
        const ssize_t c = task % channels;
        build_histogram_for_row(hist, input, h, w, channels, y, c, radius);
        for (ssize_t x = 0; x < w; ++x) {
            output[(y * w + x) * channels + c] = hist.median(target_rank);
            if (x + 1 < w) {
                slide_histogram_right(
                    hist, input, h, w, channels, y, x, c, radius);
            }
        }
    }
#endif
}

void validate_image_shape(const py::buffer_info& info, const char* op_name) {
    if (info.ndim != 2 && info.ndim != 3) {
        throw std::invalid_argument(
            std::string(op_name) + ": image must have shape (H, W) or (H, W, C)");
    }
    if (info.shape[0] <= 0 || info.shape[1] <= 0) {
        throw std::invalid_argument(
            std::string(op_name) + ": image height and width must be positive");
    }
    if (info.ndim == 3) {
        const ssize_t channels = info.shape[2];
        if (channels != 1 && channels != 3 && channels != 4) {
            throw std::invalid_argument(
                std::string(op_name) + ": channel count must be 1, 3, or 4");
        }
    }
}

void validate_ksize(ssize_t ksize, const char* op_name) {
    if (ksize <= 0 || (ksize % 2) == 0) {
        throw std::invalid_argument(
            std::string(op_name) + ": ksize must be a positive odd integer");
    }
    if (ksize > MAX_MEDIAN_FILTER_KSIZE) {
        throw std::invalid_argument(
            std::string(op_name) + ": ksize is too large");
    }
}

template <typename T>
py::array_t<T> median_filter_2d_impl(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& image,
    ssize_t ksize) {
    constexpr const char* op_name = "median_filter_2d";
    validate_ksize(ksize, op_name);

    auto input_info = image.request();
    validate_image_shape(input_info, op_name);

    std::vector<ssize_t> out_shape(input_info.shape.begin(), input_info.shape.end());
    py::array_t<T> output(out_shape);
    auto output_info = output.request();

    const ssize_t h = input_info.shape[0];
    const ssize_t w = input_info.shape[1];
    const ssize_t channels = input_info.ndim == 3 ? input_info.shape[2] : 1;

    const auto* input_ptr = static_cast<const T*>(input_info.ptr);
    auto* output_ptr = static_cast<T*>(output_info.ptr);

    py::gil_scoped_release release;
    median_filter_2d_kernel<T>(
        input_ptr, output_ptr, h, w, channels, ksize);
    return output;
}

py::array median_filter_2d_dispatch(const py::array& image, ssize_t ksize) {
    if (py::isinstance<py::array_t<uint8_t>>(image)) {
        return median_filter_2d_impl<uint8_t>(
            image.cast<py::array_t<uint8_t>>(), ksize);
    }
    if (py::isinstance<py::array_t<uint16_t>>(image)) {
        return median_filter_2d_impl<uint16_t>(
            image.cast<py::array_t<uint16_t>>(), ksize);
    }
    throw std::invalid_argument(
        "median_filter_2d: unsupported dtype; expected uint8/uint16");
}

}  // namespace

void bind_filter_ops(py::module_& m) {
    m.def("median_filter_2d",
          &median_filter_2d_dispatch,
          py::arg("image"),
          py::arg("ksize"),
          "Apply exact 2D median filtering with replicate borders.");
}
