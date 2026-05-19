#include "median_ops.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

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

template <typename T>
void median_reduce_chunk_kernel(
    const py::buffer_info& stack_info,
    py::buffer_info& out_info) {
    const auto* HNW_RESTRICT stack_ptr = static_cast<const T*>(stack_info.ptr);
    auto* HNW_RESTRICT out_ptr = static_cast<T*>(out_info.ptr);
    const ssize_t n_frames = stack_info.shape[0];
    const ssize_t plane_size = out_info.size;

#if defined(_OPENMP)
#pragma omp parallel
    {
        std::vector<T> scratch(static_cast<size_t>(n_frames));
#if HNW_ENABLE_OMP_SIMD
#pragma omp for simd schedule(static)
#else
#pragma omp for schedule(static)
#endif
        for (ssize_t idx = 0; idx < plane_size; ++idx) {
            for (ssize_t frame_idx = 0; frame_idx < n_frames; ++frame_idx) {
                scratch[static_cast<size_t>(frame_idx)] =
                    stack_ptr[frame_idx * plane_size + idx];
            }
            const ssize_t mid = n_frames / 2;
            std::nth_element(
                scratch.begin(),
                scratch.begin() + mid,
                scratch.end());
            const T high = scratch[static_cast<size_t>(mid)];
            if ((n_frames & 1) != 0) {
                out_ptr[idx] = high;
                continue;
            }
            std::nth_element(
                scratch.begin(),
                scratch.begin() + (mid - 1),
                scratch.begin() + mid);
            const T low = scratch[static_cast<size_t>(mid - 1)];
            out_ptr[idx] = static_cast<T>((low + high) * static_cast<T>(0.5));
        }
    }
#else
    std::vector<T> scratch(static_cast<size_t>(n_frames));
    for (ssize_t idx = 0; idx < plane_size; ++idx) {
        for (ssize_t frame_idx = 0; frame_idx < n_frames; ++frame_idx) {
            scratch[static_cast<size_t>(frame_idx)] =
                stack_ptr[frame_idx * plane_size + idx];
        }
        const ssize_t mid = n_frames / 2;
        std::nth_element(
            scratch.begin(),
            scratch.begin() + mid,
            scratch.end());
        const T high = scratch[static_cast<size_t>(mid)];
        if ((n_frames & 1) != 0) {
            out_ptr[idx] = high;
            continue;
        }
        std::nth_element(
            scratch.begin(),
            scratch.begin() + (mid - 1),
            scratch.begin() + mid);
        const T low = scratch[static_cast<size_t>(mid - 1)];
        out_ptr[idx] = static_cast<T>((low + high) * static_cast<T>(0.5));
    }
#endif
}

template <typename T>
py::array_t<T> median_reduce_chunk_impl(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& stack) {
    if (stack.ndim() != 3 && stack.ndim() != 4) {
        throw std::invalid_argument(
            "median_reduce_chunk: stack must have shape (N, H, W) or (N, H, W, C)");
    }
    if (stack.shape(0) <= 0) {
        throw std::invalid_argument("median_reduce_chunk: frame axis must be non-empty");
    }

    auto stack_info = stack.request();
    std::vector<ssize_t> out_shape;
    for (ssize_t dim = 1; dim < stack.ndim(); ++dim) {
        out_shape.push_back(stack.shape(dim));
    }
    py::array_t<T> out(out_shape);
    auto out_info = out.request();

    py::gil_scoped_release release;
    median_reduce_chunk_kernel<T>(stack_info, out_info);
    return out;
}

py::array median_reduce_chunk_dispatch(const py::array& stack) {
    if (py::isinstance<py::array_t<float>>(stack)) {
        return median_reduce_chunk_impl<float>(
            stack.cast<py::array_t<float>>());
    }
    if (py::isinstance<py::array_t<double>>(stack)) {
        return median_reduce_chunk_impl<double>(
            stack.cast<py::array_t<double>>());
    }
    throw std::invalid_argument(
        "median_reduce_chunk: only float32/float64 stacks are supported");
}

}  // namespace

void bind_median_ops(py::module_& m) {
    m.def("median_reduce_chunk",
          &median_reduce_chunk_dispatch,
          py::arg("stack"),
          "Reduce a stack chunk along frame axis 0 using a CPU nth_element kernel.");
}
