#include "median_reduce_ops.h"

#include "common/compat.h"

#include <pybind11/numpy.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

void launch_median_reduce_cuda_u8(const uint8_t* stack, uint8_t* output, int n_frames,
                                  int64_t plane_size);
void launch_median_reduce_cuda_u16(const uint16_t* stack, uint16_t* output, int n_frames,
                                   int64_t plane_size);
void launch_median_reduce_frames_cuda_u8(const uint8_t* const* frames, uint8_t* output,
                                         int n_frames, int64_t plane_size);
void launch_median_reduce_frames_cuda_u16(const uint16_t* const* frames, uint16_t* output,
                                          int n_frames, int64_t plane_size);

namespace {

template <typename T, typename Launcher>
py::array_t<T>
median_reduce_cuda_impl(const py::array_t<T, py::array::c_style | py::array::forcecast>& stack,
                        Launcher launch) {
    if (stack.ndim() != 3 && stack.ndim() != 4) {
        throw std::invalid_argument(
            "median_reduce_chunk_cuda: stack must have shape (N, H, W) or (N, H, W, C)");
    }
    const ssize_t n_frames = stack.shape(0);
    if (n_frames < 1 || n_frames > 128) {
        throw std::invalid_argument("median_reduce_chunk_cuda: frame count must be in [1, 128]");
    }
    std::vector<ssize_t> output_shape;
    for (ssize_t dim = 1; dim < stack.ndim(); ++dim) {
        output_shape.push_back(stack.shape(dim));
    }
    py::array_t<T> output(output_shape);
    const int64_t plane_size = static_cast<int64_t>(output.size());
    if (plane_size > 0) {
        py::gil_scoped_release release;
        launch(stack.data(), output.mutable_data(), static_cast<int>(n_frames), plane_size);
    }
    return output;
}

py::array median_reduce_cuda_dispatch(const py::array& stack) {
    if (py::isinstance<py::array_t<uint8_t>>(stack)) {
        return median_reduce_cuda_impl(
            stack.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(),
            launch_median_reduce_cuda_u8);
    }
    if (py::isinstance<py::array_t<uint16_t>>(stack)) {
        return median_reduce_cuda_impl(
            stack.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(),
            launch_median_reduce_cuda_u16);
    }
    throw std::invalid_argument("median_reduce_chunk_cuda: expected uint8 or uint16 stack");
}

template <typename T, typename Launcher>
py::array_t<T> median_reduce_frames_cuda_impl(const py::sequence& frames, Launcher launch) {
    const ssize_t n_frames = py::len(frames);
    if (n_frames < 1 || n_frames > 128) {
        throw std::invalid_argument(
            "median_reduce_chunk_cuda_frames: frame count must be in [1, 128]");
    }
    auto first = frames[0].cast<py::array_t<T, py::array::c_style | py::array::forcecast>>();
    if (first.ndim() != 2 && first.ndim() != 3) {
        throw std::invalid_argument("median_reduce_chunk_cuda_frames: frames must be 2D or 3D");
    }
    std::vector<ssize_t> output_shape;
    for (ssize_t dim = 0; dim < first.ndim(); ++dim) {
        output_shape.push_back(first.shape(dim));
    }
    std::vector<py::array_t<T, py::array::c_style | py::array::forcecast>> arrays;
    std::vector<const T*> pointers;
    arrays.reserve(static_cast<size_t>(n_frames));
    pointers.reserve(static_cast<size_t>(n_frames));
    for (ssize_t frame = 0; frame < n_frames; ++frame) {
        py::array item = frames[frame].cast<py::array>();
        if (!item.dtype().is(py::dtype::of<T>()) || item.ndim() != first.ndim()) {
            throw std::invalid_argument(
                "median_reduce_chunk_cuda_frames: frame dtype or rank mismatch");
        }
        for (ssize_t dim = 0; dim < first.ndim(); ++dim) {
            if (item.shape(dim) != first.shape(dim)) {
                throw std::invalid_argument(
                    "median_reduce_chunk_cuda_frames: frame shape mismatch");
            }
        }
        arrays.push_back(item.cast<py::array_t<T, py::array::c_style | py::array::forcecast>>());
        pointers.push_back(arrays.back().data());
    }
    py::array_t<T> output(output_shape);
    const int64_t plane_size = static_cast<int64_t>(output.size());
    if (plane_size > 0) {
        py::gil_scoped_release release;
        launch(pointers.data(), output.mutable_data(), static_cast<int>(n_frames), plane_size);
    }
    return output;
}

py::array median_reduce_frames_cuda_dispatch(const py::sequence& frames) {
    if (py::len(frames) < 1) {
        throw std::invalid_argument(
            "median_reduce_chunk_cuda_frames: frame count must be positive");
    }
    const py::array first = frames[0].cast<py::array>();
    if (first.dtype().is(py::dtype::of<uint8_t>())) {
        return median_reduce_frames_cuda_impl<uint8_t>(frames, launch_median_reduce_frames_cuda_u8);
    }
    if (first.dtype().is(py::dtype::of<uint16_t>())) {
        return median_reduce_frames_cuda_impl<uint16_t>(frames,
                                                        launch_median_reduce_frames_cuda_u16);
    }
    throw std::invalid_argument("median_reduce_chunk_cuda_frames: expected uint8 or uint16 frames");
}

} // namespace

void bind_median_reduce_cuda_ops(py::module_& m) {
    m.def("median_reduce_chunk_cuda", &median_reduce_cuda_dispatch, py::arg("stack"),
          "Exact CUDA median across up to 128 integer frames.");
    m.def("median_reduce_chunk_cuda_frames", &median_reduce_frames_cuda_dispatch, py::arg("frames"),
          "Exact CUDA median from per-frame chunks.");
}
