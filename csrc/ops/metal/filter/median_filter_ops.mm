#include "median_filter_ops.h"

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

namespace py = pybind11;

namespace {

constexpr const char* kContext = "median_filter_2d_metal";

struct MedianParams {
    uint32_t height;
    uint32_t width;
};

void launch_median_filter_metal(const uint16_t* input, uint16_t* output, const int height,
                                const int width) {
    @autoreleasepool {
        auto& workspace = hnw::metal::HostIOWorkspace::current();
        workspace.begin_operation("median_filter_2d");
        try {
            const uint32_t count = static_cast<uint32_t>(static_cast<uint64_t>(height) * width);
            const size_t bytes = static_cast<size_t>(count) * sizeof(uint16_t);
            id<MTLBuffer> source = workspace.buffer(bytes, "median_filter_2d_metal source");
            id<MTLBuffer> result = workspace.buffer(bytes, "median_filter_2d_metal result");
            std::memcpy(source.contents, input, bytes);

            const MedianParams params{static_cast<uint32_t>(height), static_cast<uint32_t>(width)};
            id<MTLCommandBuffer> command =
                hnw::metal::new_command_buffer(workspace.command_queue(), kContext);
            id<MTLComputePipelineState> pipeline = workspace.pipeline("median_filter_u16_13");
            id<MTLComputeCommandEncoder> encoder =
                hnw::metal::begin_encoder(command, pipeline, kContext);
            [encoder setBuffer:source offset:0 atIndex:0];
            [encoder setBuffer:result offset:0 atIndex:1];
            [encoder setBytes:&params length:sizeof(params) atIndex:2];
            hnw::metal::dispatch_1d(encoder, pipeline, count);
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            hnw::metal::throw_if_command_failed(command, kContext);

            std::memcpy(output, result.contents, bytes);
            workspace.finish_operation();
        } catch (...) {
            workspace.reset_after_error();
            throw;
        }
    }
}

py::array_t<uint16_t> median_filter_2d_metal(const py::array& source, const int ksize) {
    if (!source.dtype().is(py::dtype::of<uint16_t>()) || source.ndim() != 2 ||
        source.shape(0) <= 0 || source.shape(1) <= 0) {
        throw std::invalid_argument("median_filter_2d_metal: expected a non-empty 2D uint16 image");
    }
    if (ksize != 13) {
        throw std::invalid_argument("median_filter_2d_metal: only ksize=13 is supported");
    }
    if (source.shape(0) > std::numeric_limits<int>::max() - 13 ||
        source.shape(1) > std::numeric_limits<int>::max() - 13 ||
        static_cast<uint64_t>(source.shape(0)) * source.shape(1) >
            std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("median_filter_2d_metal: image is too large");
    }

    const auto image =
        source.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>();
    py::array_t<uint16_t> result({image.shape(0), image.shape(1)});
    const uint16_t* input = image.data();
    uint16_t* output = result.mutable_data();
    {
        py::gil_scoped_release release;
        launch_median_filter_metal(input, output, static_cast<int>(image.shape(0)),
                                   static_cast<int>(image.shape(1)));
    }
    return result;
}

} // namespace

void bind_median_filter_metal_ops(py::module_& m) {
    m.def("median_filter_2d_metal", &median_filter_2d_metal, py::arg("image"), py::arg("ksize"),
          "Exact uint16 13x13 median filter on Metal; nearest border samples.");
}
