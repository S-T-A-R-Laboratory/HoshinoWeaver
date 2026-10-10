#include "median_reduce_ops.h"

#include "common/compat.h"
#include "common/metal_dispatch.h"
#include "common/metal_host_io_workspace.h"

#include <pybind11/numpy.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace py = pybind11;

namespace {

constexpr const char* kContext = "median_reduce_chunk_metal";

struct MedianStackParams {
    uint32_t n_frames;
    uint32_t plane_size;
    uint32_t top_shift;
};

template <typename T, typename FillInput>
void launch_median_reduce_metal(T* output_host, const int n_frames, const int64_t plane_size,
                                FillInput fill_input) {
    @autoreleasepool {
        auto& workspace = hnw::metal::HostIOWorkspace::current();
        workspace.begin_operation("median_reduce_chunk");
        try {
            const size_t input_bytes = static_cast<size_t>(n_frames) * plane_size * sizeof(T);
            const size_t output_bytes = static_cast<size_t>(plane_size) * sizeof(T);
            id<MTLBuffer> source = workspace.buffer(input_bytes, "median_reduce_chunk_metal input");
            id<MTLBuffer> result =
                workspace.buffer(output_bytes, "median_reduce_chunk_metal output");
            fill_input(static_cast<T*>(source.contents), output_bytes);

            const MedianStackParams params{
                static_cast<uint32_t>(n_frames),
                static_cast<uint32_t>(plane_size),
                std::is_same_v<T, uint8_t> ? 4u : 12u,
            };
            id<MTLCommandBuffer> command =
                hnw::metal::new_command_buffer(workspace.command_queue(), kContext);
            const char* function_name =
                std::is_same_v<T, uint8_t> ? "median_reduce_stack_u8" : "median_reduce_stack_u16";
            id<MTLComputePipelineState> pipeline = workspace.pipeline(function_name);
            id<MTLComputeCommandEncoder> encoder =
                hnw::metal::begin_encoder(command, pipeline, kContext);
            [encoder setBuffer:source offset:0 atIndex:0];
            [encoder setBuffer:result offset:0 atIndex:1];
            [encoder setBytes:&params length:sizeof(params) atIndex:2];
            hnw::metal::dispatch_1d(encoder, pipeline, static_cast<uint32_t>(plane_size));
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            hnw::metal::throw_if_command_failed(command, kContext);

            std::memcpy(output_host, result.contents, output_bytes);
            workspace.finish_operation();
        } catch (...) {
            workspace.reset_after_error();
            throw;
        }
    }
}

void validate_count(const ssize_t n_frames, const int64_t plane_size) {
    if (n_frames < 1 || n_frames > 128 || plane_size < 0 ||
        static_cast<uint64_t>(plane_size) >
            std::numeric_limits<uint32_t>::max() / static_cast<uint64_t>(n_frames)) {
        throw std::invalid_argument("median_reduce_chunk_metal: invalid stack dimensions");
    }
}

template <typename T> py::array_t<T> reduce_stack(const py::array& source) {
    if (source.ndim() != 3 && source.ndim() != 4) {
        throw std::invalid_argument("median_reduce_chunk_metal: expected a 3D or 4D stack");
    }
    const auto stack = source.cast<py::array_t<T, py::array::c_style | py::array::forcecast>>();
    if (stack.shape(0) < 1 || stack.shape(0) > 128) {
        throw std::invalid_argument("median_reduce_chunk_metal: expected 1-128 frames");
    }
    const int64_t plane_size = static_cast<int64_t>(stack.size() / stack.shape(0));
    validate_count(stack.shape(0), plane_size);
    std::vector<ssize_t> output_shape;
    for (ssize_t dim = 1; dim < stack.ndim(); ++dim) {
        output_shape.push_back(stack.shape(dim));
    }
    py::array_t<T> output(output_shape);
    if (plane_size > 0) {
        const T* input_ptr = stack.data();
        T* output_ptr = output.mutable_data();
        py::gil_scoped_release release;
        launch_median_reduce_metal(
            output_ptr, static_cast<int>(stack.shape(0)), plane_size,
            [input_ptr, n_frames = stack.shape(0)](T* buffer, size_t frame_bytes) {
                std::memcpy(buffer, input_ptr, static_cast<size_t>(n_frames) * frame_bytes);
            });
    }
    return output;
}

py::array reduce_stack_dispatch(const py::array& stack) {
    if (stack.dtype().is(py::dtype::of<uint8_t>())) {
        return reduce_stack<uint8_t>(stack);
    }
    if (stack.dtype().is(py::dtype::of<uint16_t>())) {
        return reduce_stack<uint16_t>(stack);
    }
    throw std::invalid_argument("median_reduce_chunk_metal: expected uint8 or uint16 stack");
}

template <typename T> py::array_t<T> reduce_frames(const py::sequence& frames) {
    const ssize_t n_frames = py::len(frames);
    if (n_frames < 1 || n_frames > 128) {
        throw std::invalid_argument("median_reduce_chunk_metal_frames: expected 1-128 frames");
    }
    const auto first = frames[0].cast<py::array_t<T, py::array::c_style | py::array::forcecast>>();
    if (first.ndim() != 2 && first.ndim() != 3) {
        throw std::invalid_argument("median_reduce_chunk_metal_frames: expected 2D/3D frames");
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
        const py::array item = frames[frame].cast<py::array>();
        if (!item.dtype().is(py::dtype::of<T>()) || item.ndim() != first.ndim()) {
            throw std::invalid_argument(
                "median_reduce_chunk_metal_frames: frame dtype/rank mismatch");
        }
        for (ssize_t dim = 0; dim < first.ndim(); ++dim) {
            if (item.shape(dim) != first.shape(dim)) {
                throw std::invalid_argument(
                    "median_reduce_chunk_metal_frames: frame shape mismatch");
            }
        }
        arrays.push_back(item.cast<py::array_t<T, py::array::c_style | py::array::forcecast>>());
        pointers.push_back(arrays.back().data());
    }
    const int64_t plane_size = static_cast<int64_t>(first.size());
    validate_count(n_frames, plane_size);
    py::array_t<T> output(output_shape);
    if (plane_size > 0) {
        T* output_ptr = output.mutable_data();
        py::gil_scoped_release release;
        launch_median_reduce_metal(
            output_ptr, static_cast<int>(n_frames), plane_size,
            [&pointers, n_frames, plane_size](T* buffer, size_t frame_bytes) {
                for (ssize_t frame = 0; frame < n_frames; ++frame) {
                    std::memcpy(buffer + static_cast<size_t>(frame) * plane_size,
                                pointers[static_cast<size_t>(frame)], frame_bytes);
                }
            });
    }
    return output;
}

py::array reduce_frames_dispatch(const py::sequence& frames) {
    if (py::len(frames) < 1) {
        throw std::invalid_argument("median_reduce_chunk_metal_frames: no frames");
    }
    const py::array first = frames[0].cast<py::array>();
    if (first.dtype().is(py::dtype::of<uint8_t>())) {
        return reduce_frames<uint8_t>(frames);
    }
    if (first.dtype().is(py::dtype::of<uint16_t>())) {
        return reduce_frames<uint16_t>(frames);
    }
    throw std::invalid_argument("median_reduce_chunk_metal_frames: expected uint8/uint16 frames");
}

} // namespace

void bind_median_reduce_metal_ops(py::module_& m) {
    m.def("median_reduce_chunk_metal", &reduce_stack_dispatch, py::arg("stack"),
          "Exact Metal median across up to 128 integer frames.");
    m.def("median_reduce_chunk_metal_frames", &reduce_frames_dispatch, py::arg("frames"),
          "Exact Metal median from per-frame chunks.");
}
