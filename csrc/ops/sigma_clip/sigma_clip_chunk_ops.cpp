#include "sigma_clip_chunk_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <pybind11/numpy.h>

#include "common/compat.h"

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

template <typename T>
inline bool is_pixel_zero_rgb_chunk(const T* HNW_RESTRICT ptr,
                                    ssize_t base, ssize_t channels) {
    for (ssize_t c = 0; c < channels && c < 3; ++c) {
        if (ptr[base + c] != static_cast<T>(0)) return false;
    }
    return true;
}

template <typename T>
void sigma_clip_iterative_chunk_kernel(
    const T* HNW_RESTRICT stack,
    ssize_t n_frames,
    ssize_t plane_size,
    const double* HNW_RESTRICT total_sum,
    const double* HNW_RESTRICT total_sq,
    const double* HNW_RESTRICT total_n,
    double rej_high,
    double rej_low,
    int max_iter,
    const uint8_t* HNW_RESTRICT mask,  // NULL or (n_frames * plane_size)
    double* HNW_RESTRICT out_sum,
    double* HNW_RESTRICT out_sq,
    double* HNW_RESTRICT out_n,
    bool skip_zero_rgb = false,
    ssize_t channels = 1) {

    // Working arrays
    std::vector<double> cur_sum(static_cast<size_t>(plane_size));
    std::vector<double> cur_sq(static_cast<size_t>(plane_size));
    std::vector<double> cur_n(static_cast<size_t>(plane_size));
    std::vector<double> low(static_cast<size_t>(plane_size));
    std::vector<double> high(static_cast<size_t>(plane_size));
    std::vector<uint8_t> converged(static_cast<size_t>(plane_size), 0);

    // Initialize current stats from total
    std::copy(total_sum, total_sum + plane_size, cur_sum.data());
    std::copy(total_sq, total_sq + plane_size, cur_sq.data());
    std::copy(total_n, total_n + plane_size, cur_n.data());

    // Dtype range for threshold clamping
    constexpr double dtype_min = 0.0;
    constexpr double dtype_max = static_cast<double>(std::numeric_limits<T>::max());

    for (int iter = 0; iter < max_iter; ++iter) {
        // 1. Compute thresholds from current accepted stats.
        // 每个 idx 独立，按像素并行不会产生写冲突。
#if defined(_OPENMP) && HNW_ENABLE_OMP_SIMD
#pragma omp parallel for simd schedule(static)
#elif defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (ssize_t idx = 0; idx < plane_size; ++idx) {
            if (converged[idx]) continue;
            const double n = cur_n[idx];
            if (n <= 1.0) {
                converged[idx] = 1;
                continue;
            }
            const double mu = cur_sum[idx] / n;
            const double var = (cur_sq[idx] - cur_sum[idx] * cur_sum[idx] / n)
                               / (n - 1.0);
            const double std_val = std::sqrt(std::fmax(var, 0.0));
            high[idx] = std::fmin(std::floor(mu + std_val * rej_high), dtype_max);
            low[idx] = std::fmax(std::ceil(mu - std_val * rej_low), dtype_min);
        }

        // 2. Rejected accumulators (allocated per iteration to ensure zeroed)
        std::vector<double> rej_sum(static_cast<size_t>(plane_size), 0.0);
        std::vector<double> rej_sq(static_cast<size_t>(plane_size), 0.0);
        std::vector<double> rej_n(static_cast<size_t>(plane_size), 0.0);

        // 3. Scan all frames. 按输出像素分片，每个线程只写自己的 rej_* 区间。
        if (skip_zero_rgb && channels >= 3) {
            const ssize_t spatial = plane_size / channels;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (ssize_t px = 0; px < spatial; ++px) {
                const ssize_t base = px * channels;
                for (ssize_t f = 0; f < n_frames; ++f) {
                    const T* HNW_RESTRICT frame_row = stack + f * plane_size;
                    const uint8_t* HNW_RESTRICT mask_row =
                        mask ? mask + f * plane_size : nullptr;
                    if (is_pixel_zero_rgb_chunk(frame_row, base, channels))
                        continue;
#if defined(HNW_ENABLE_OMP_SIMD) && HNW_ENABLE_OMP_SIMD
#pragma omp simd
#endif
                    for (ssize_t c = 0; c < channels; ++c) {
                        const ssize_t idx = base + c;
                        if (converged[idx]) continue;
                        if (mask_row && !mask_row[idx]) continue;
                        const double val = static_cast<double>(frame_row[idx]);
                        if (val < low[idx] || val > high[idx]) {
                            rej_sum[idx] += val;
                            rej_sq[idx] += val * val;
                            rej_n[idx] += 1.0;
                        }
                    }
                }
            }
        } else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (ssize_t idx = 0; idx < plane_size; ++idx) {
                if (converged[idx]) continue;
                for (ssize_t f = 0; f < n_frames; ++f) {
                    const T* HNW_RESTRICT frame_row = stack + f * plane_size;
                    const uint8_t* HNW_RESTRICT mask_row =
                        mask ? mask + f * plane_size : nullptr;
                    if (mask_row && !mask_row[idx]) continue;
                    const double val = static_cast<double>(frame_row[idx]);
                    if (val < low[idx] || val > high[idx]) {
                        rej_sum[idx] += val;
                        rej_sq[idx] += val * val;
                        rej_n[idx] += 1.0;
                    }
                }
            }
        }

        // 4. Update accepted stats + convergence check
        int changed_count = 0;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) reduction(+:changed_count)
#endif
        for (ssize_t idx = 0; idx < plane_size; ++idx) {
            if (converged[idx]) continue;
            const double new_n = total_n[idx] - rej_n[idx];
            const double new_sum = total_sum[idx] - rej_sum[idx];
            const double new_sq = total_sq[idx] - rej_sq[idx];
            if (new_n == cur_n[idx] && new_sum == cur_sum[idx] && new_sq == cur_sq[idx]) {
                converged[idx] = 1;
            } else if (new_n <= 0.0) {
                // All frames rejected — fall back to total stats
                cur_sum[idx] = total_sum[idx];
                cur_sq[idx] = total_sq[idx];
                cur_n[idx] = total_n[idx];
                converged[idx] = 1;
            } else {
                cur_sum[idx] = new_sum;
                cur_sq[idx] = new_sq;
                cur_n[idx] = new_n;
                changed_count += 1;
            }
        }
        if (changed_count == 0) break;
    }

    // Output
    std::copy(cur_sum.data(), cur_sum.data() + plane_size, out_sum);
    std::copy(cur_sq.data(), cur_sq.data() + plane_size, out_sq);
    std::copy(cur_n.data(), cur_n.data() + plane_size, out_n);
}

py::tuple sigma_clip_iterative_chunk_dispatch(
    const py::array& stack,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& total_sum,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& total_sq,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& total_n,
    double rej_high,
    double rej_low,
    int max_iter,
    const py::object& mask_obj,
    bool skip_zero_rgb,
    ssize_t channels) {

    // Validate stack shape: must be 2D (n_frames, plane_size)
    if (stack.ndim() != 2) {
        throw std::invalid_argument(
            "sigma_clip_iterative_chunk: stack must be 2D (n_frames, plane_size)");
    }
    const ssize_t n_frames = stack.shape(0);
    const ssize_t plane_size = stack.shape(1);

    if (n_frames <= 0) {
        throw std::invalid_argument(
            "sigma_clip_iterative_chunk: n_frames must be > 0");
    }
    if (total_sum.size() != plane_size ||
        total_sq.size() != plane_size ||
        total_n.size() != plane_size) {
        throw std::invalid_argument(
            "sigma_clip_iterative_chunk: total stats size must match plane_size");
    }

    // Parse optional mask
    const uint8_t* mask_ptr = nullptr;
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask_arr;
    if (!mask_obj.is_none()) {
        mask_arr = mask_obj.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
        if (mask_arr.ndim() != 2 ||
            mask_arr.shape(0) != n_frames ||
            mask_arr.shape(1) != plane_size) {
            throw std::invalid_argument(
                "sigma_clip_iterative_chunk: mask must have shape (n_frames, plane_size)");
        }
        mask_ptr = static_cast<const uint8_t*>(mask_arr.request().ptr);
    }

    // Allocate output arrays
    py::array_t<double> out_sum({plane_size});
    py::array_t<double> out_sq({plane_size});
    py::array_t<double> out_n({plane_size});

    auto sum_info = total_sum.request();
    auto sq_info = total_sq.request();
    auto n_info = total_n.request();
    auto out_sum_info = out_sum.request();
    auto out_sq_info = out_sq.request();
    auto out_n_info = out_n.request();

    // Dispatch on stack dtype
    if (py::isinstance<py::array_t<uint8_t>>(stack)) {
        auto stack_arr = stack.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
        auto stack_info = stack_arr.request();
        py::gil_scoped_release release;
        sigma_clip_iterative_chunk_kernel<uint8_t>(
            static_cast<const uint8_t*>(stack_info.ptr),
            n_frames, plane_size,
            static_cast<const double*>(sum_info.ptr),
            static_cast<const double*>(sq_info.ptr),
            static_cast<const double*>(n_info.ptr),
            rej_high, rej_low, max_iter,
            mask_ptr,
            static_cast<double*>(out_sum_info.ptr),
            static_cast<double*>(out_sq_info.ptr),
            static_cast<double*>(out_n_info.ptr),
            skip_zero_rgb, channels);
    } else if (py::isinstance<py::array_t<uint16_t>>(stack)) {
        auto stack_arr = stack.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>();
        auto stack_info = stack_arr.request();
        py::gil_scoped_release release;
        sigma_clip_iterative_chunk_kernel<uint16_t>(
            static_cast<const uint16_t*>(stack_info.ptr),
            n_frames, plane_size,
            static_cast<const double*>(sum_info.ptr),
            static_cast<const double*>(sq_info.ptr),
            static_cast<const double*>(n_info.ptr),
            rej_high, rej_low, max_iter,
            mask_ptr,
            static_cast<double*>(out_sum_info.ptr),
            static_cast<double*>(out_sq_info.ptr),
            static_cast<double*>(out_n_info.ptr),
            skip_zero_rgb, channels);
    } else {
        throw std::invalid_argument(
            "sigma_clip_iterative_chunk: unsupported stack dtype; expected uint8/uint16");
    }

    return py::make_tuple(out_sum, out_sq, out_n);
}

// --- Fused variant: computes mean FGP internally then iterative clip ---

template <typename T>
void sigma_clip_fused_chunk_kernel(
    const T* HNW_RESTRICT stack,
    ssize_t n_frames,
    ssize_t plane_size,
    double rej_high,
    double rej_low,
    int max_iter,
    const uint8_t* HNW_RESTRICT mask,  // NULL or (n_frames * plane_size)
    double* HNW_RESTRICT out_sum,
    double* HNW_RESTRICT out_sq,
    double* HNW_RESTRICT out_n,
    bool skip_zero_rgb = false,
    ssize_t channels = 1) {

    // Phase 1: Compute total FGP from stack (respecting mask)
    std::vector<double> total_sum(static_cast<size_t>(plane_size), 0.0);
    std::vector<double> total_sq(static_cast<size_t>(plane_size), 0.0);
    std::vector<double> total_n(static_cast<size_t>(plane_size), 0.0);

    if (skip_zero_rgb && channels >= 3) {
        const ssize_t spatial = plane_size / channels;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (ssize_t px = 0; px < spatial; ++px) {
            const ssize_t base = px * channels;
            for (ssize_t f = 0; f < n_frames; ++f) {
                const T* HNW_RESTRICT frame_row = stack + f * plane_size;
                const uint8_t* HNW_RESTRICT mask_row =
                    mask ? mask + f * plane_size : nullptr;
                if (is_pixel_zero_rgb_chunk(frame_row, base, channels))
                    continue;
#if defined(HNW_ENABLE_OMP_SIMD) && HNW_ENABLE_OMP_SIMD
#pragma omp simd
#endif
                for (ssize_t c = 0; c < channels; ++c) {
                    const ssize_t idx = base + c;
                    if (mask_row && !mask_row[idx]) continue;
                    const double val = static_cast<double>(frame_row[idx]);
                    total_sum[idx] += val;
                    total_sq[idx] += val * val;
                    total_n[idx] += 1.0;
                }
            }
        }
    } else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (ssize_t idx = 0; idx < plane_size; ++idx) {
            for (ssize_t f = 0; f < n_frames; ++f) {
                const T* HNW_RESTRICT frame_row = stack + f * plane_size;
                const uint8_t* HNW_RESTRICT mask_row =
                    mask ? mask + f * plane_size : nullptr;
                if (mask_row && !mask_row[idx]) continue;
                const double val = static_cast<double>(frame_row[idx]);
                total_sum[idx] += val;
                total_sq[idx] += val * val;
                total_n[idx] += 1.0;
            }
        }
    }

    // Phase 2: Iterative sigma clip (reuse existing kernel logic)
    sigma_clip_iterative_chunk_kernel<T>(
        stack, n_frames, plane_size,
        total_sum.data(), total_sq.data(), total_n.data(),
        rej_high, rej_low, max_iter,
        mask,
        out_sum, out_sq, out_n,
        skip_zero_rgb, channels);
}

py::tuple sigma_clip_fused_chunk_dispatch(
    const py::array& stack,
    double rej_high,
    double rej_low,
    int max_iter,
    const py::object& mask_obj,
    bool skip_zero_rgb,
    ssize_t channels) {

    if (stack.ndim() != 2) {
        throw std::invalid_argument(
            "sigma_clip_fused_chunk: stack must be 2D (n_frames, plane_size)");
    }
    const ssize_t n_frames = stack.shape(0);
    const ssize_t plane_size = stack.shape(1);

    if (n_frames <= 0) {
        throw std::invalid_argument(
            "sigma_clip_fused_chunk: n_frames must be > 0");
    }

    // Parse optional mask
    const uint8_t* mask_ptr = nullptr;
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask_arr;
    if (!mask_obj.is_none()) {
        mask_arr = mask_obj.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
        if (mask_arr.ndim() != 2 ||
            mask_arr.shape(0) != n_frames ||
            mask_arr.shape(1) != plane_size) {
            throw std::invalid_argument(
                "sigma_clip_fused_chunk: mask must have shape (n_frames, plane_size)");
        }
        mask_ptr = static_cast<const uint8_t*>(mask_arr.request().ptr);
    }

    py::array_t<double> out_sum({plane_size});
    py::array_t<double> out_sq({plane_size});
    py::array_t<double> out_n({plane_size});

    auto out_sum_info = out_sum.request();
    auto out_sq_info = out_sq.request();
    auto out_n_info = out_n.request();

    if (py::isinstance<py::array_t<uint8_t>>(stack)) {
        auto stack_arr = stack.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
        auto stack_info = stack_arr.request();
        py::gil_scoped_release release;
        sigma_clip_fused_chunk_kernel<uint8_t>(
            static_cast<const uint8_t*>(stack_info.ptr),
            n_frames, plane_size,
            rej_high, rej_low, max_iter,
            mask_ptr,
            static_cast<double*>(out_sum_info.ptr),
            static_cast<double*>(out_sq_info.ptr),
            static_cast<double*>(out_n_info.ptr),
            skip_zero_rgb, channels);
    } else if (py::isinstance<py::array_t<uint16_t>>(stack)) {
        auto stack_arr = stack.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>();
        auto stack_info = stack_arr.request();
        py::gil_scoped_release release;
        sigma_clip_fused_chunk_kernel<uint16_t>(
            static_cast<const uint16_t*>(stack_info.ptr),
            n_frames, plane_size,
            rej_high, rej_low, max_iter,
            mask_ptr,
            static_cast<double*>(out_sum_info.ptr),
            static_cast<double*>(out_sq_info.ptr),
            static_cast<double*>(out_n_info.ptr),
            skip_zero_rgb, channels);
    } else {
        throw std::invalid_argument(
            "sigma_clip_fused_chunk: unsupported stack dtype; expected uint8/uint16");
    }

    return py::make_tuple(out_sum, out_sq, out_n);
}

}  // namespace

void bind_sigma_clip_chunk_ops(py::module_& m) {
    m.def("sigma_clip_iterative_chunk",
          &sigma_clip_iterative_chunk_dispatch,
          py::arg("stack"),
          py::arg("total_sum"),
          py::arg("total_sq"),
          py::arg("total_n"),
          py::arg("rej_high"),
          py::arg("rej_low"),
          py::arg("max_iter"),
          py::arg("mask") = py::none(),
          py::arg("skip_zero_rgb") = false,
          py::arg("channels") = static_cast<ssize_t>(1),
          "Iterative sigma clipping on a chunk stack. Returns (accepted_sum, accepted_sq, accepted_n).");

    m.def("sigma_clip_fused_chunk",
          &sigma_clip_fused_chunk_dispatch,
          py::arg("stack"),
          py::arg("rej_high"),
          py::arg("rej_low"),
          py::arg("max_iter"),
          py::arg("mask") = py::none(),
          py::arg("skip_zero_rgb") = false,
          py::arg("channels") = static_cast<ssize_t>(1),
          "Fused mean + iterative sigma clipping on a chunk stack. Returns (accepted_sum, accepted_sq, accepted_n).");
}
