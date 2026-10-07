#include "star_mask_ops.h"

#include "common/compat.h"
#include "common/gaussian_kernel.h"
#include "common/metal_dispatch.h"
#include "common/metal_error.h"
#include "common/metal_host_io_workspace.h"
#include "star_mask_dog_metal_internal.h"
#include "star_shrink_metal_params.h"

#include <pybind11/numpy.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using hnw::metal::star_shrink::compute_threshold;
using hnw::metal::star_shrink::encode_gaussian;
using hnw::metal::star_shrink::encode_morphology;
using hnw::metal::star_shrink::MaskParams;
using hnw::metal::star_shrink::validate_common;

constexpr const char* kContext = "star_mask_dog_metal";

template <typename T>
void launch_star_mask_dog_metal(const T* image_host, uint8_t* mask_host, const int height,
                                const int width, const int channels,
                                const std::vector<float>& small_kernel,
                                const std::vector<float>& large_kernel, const float threshold_ratio,
                                const int open_ksize, const int dilate_ksize) {
    @autoreleasepool {
        auto& workspace = hnw::metal::HostIOWorkspace::current();
        workspace.begin_operation("star_mask_dog");
        try {
            const uint32_t plane_size =
                static_cast<uint32_t>(static_cast<uint64_t>(height) * static_cast<uint64_t>(width));
            const uint32_t total = static_cast<uint32_t>(static_cast<uint64_t>(plane_size) *
                                                         static_cast<uint64_t>(channels));
            const size_t image_bytes = static_cast<size_t>(total) * sizeof(T);
            const size_t plane_float_bytes = static_cast<size_t>(plane_size) * sizeof(float);
            const size_t mask_bytes = static_cast<size_t>(plane_size);
            const size_t small_weight_bytes = small_kernel.size() * sizeof(float);
            const size_t large_weight_bytes = large_kernel.size() * sizeof(float);

            id<MTLBuffer> image = workspace.buffer(image_bytes, "star_mask_dog_metal image");
            id<MTLBuffer> gray = workspace.buffer(plane_float_bytes, "star_mask_dog_metal gray");
            id<MTLBuffer> tmp = workspace.buffer(plane_float_bytes, "star_mask_dog_metal tmp");
            id<MTLBuffer> blur_small =
                workspace.buffer(plane_float_bytes, "star_mask_dog_metal blur_small");
            id<MTLBuffer> blur_large =
                workspace.buffer(plane_float_bytes, "star_mask_dog_metal blur_large");
            id<MTLBuffer> dog = workspace.buffer(plane_float_bytes, "star_mask_dog_metal dog");
            id<MTLBuffer> small_weights =
                workspace.buffer(small_weight_bytes, "star_mask_dog_metal small_weights");
            id<MTLBuffer> large_weights =
                workspace.buffer(large_weight_bytes, "star_mask_dog_metal large_weights");
            id<MTLBuffer> mask = workspace.buffer(mask_bytes, "star_mask_dog_metal mask");
            id<MTLBuffer> scratch = workspace.buffer(mask_bytes, "star_mask_dog_metal scratch");

            std::memcpy(image.contents, image_host, image_bytes);
            std::memcpy(small_weights.contents, small_kernel.data(), small_weight_bytes);
            std::memcpy(large_weights.contents, large_kernel.data(), large_weight_bytes);

            MaskParams params{
                static_cast<uint32_t>(height),
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(channels),
                0,
                0.0f,
            };
            const int small_radius = static_cast<int>(small_kernel.size() / 2);
            const int large_radius = static_cast<int>(large_kernel.size() / 2);

            // First pass stops at the DoG plane so the host can derive the threshold.
            id<MTLCommandBuffer> dog_pass =
                hnw::metal::new_command_buffer(workspace.command_queue(), kContext);
            {
                const std::string name =
                    std::string("star_mask_gray_") + (std::is_same_v<T, uint8_t> ? "u8" : "u16");
                id<MTLComputePipelineState> pipeline = workspace.pipeline(name.c_str());
                id<MTLComputeCommandEncoder> encoder =
                    hnw::metal::begin_encoder(dog_pass, pipeline, kContext);
                [encoder setBuffer:image offset:0 atIndex:0];
                [encoder setBuffer:gray offset:0 atIndex:1];
                [encoder setBytes:&params length:sizeof(params) atIndex:2];
                hnw::metal::dispatch_1d(encoder, pipeline, plane_size);
                [encoder endEncoding];
            }
            encode_gaussian(dog_pass, workspace, gray, tmp, blur_small, small_weights, params,
                            small_radius, plane_size, kContext);
            encode_gaussian(dog_pass, workspace, gray, tmp, blur_large, large_weights, params,
                            large_radius, plane_size, kContext);
            {
                id<MTLComputePipelineState> pipeline = workspace.pipeline("star_mask_dog_diff");
                id<MTLComputeCommandEncoder> encoder =
                    hnw::metal::begin_encoder(dog_pass, pipeline, kContext);
                [encoder setBuffer:blur_small offset:0 atIndex:0];
                [encoder setBuffer:blur_large offset:0 atIndex:1];
                [encoder setBuffer:dog offset:0 atIndex:2];
                [encoder setBytes:&params length:sizeof(params) atIndex:3];
                hnw::metal::dispatch_1d(encoder, pipeline, plane_size);
                [encoder endEncoding];
            }
            [dog_pass commit];
            [dog_pass waitUntilCompleted];
            hnw::metal::throw_if_command_failed(dog_pass, kContext);

            params.threshold = compute_threshold(static_cast<const float*>(dog.contents),
                                                 plane_size, threshold_ratio);

            id<MTLCommandBuffer> mask_pass =
                hnw::metal::new_command_buffer(workspace.command_queue(), kContext);
            {
                id<MTLComputePipelineState> pipeline = workspace.pipeline("star_mask_threshold");
                id<MTLComputeCommandEncoder> encoder =
                    hnw::metal::begin_encoder(mask_pass, pipeline, kContext);
                [encoder setBuffer:dog offset:0 atIndex:0];
                [encoder setBuffer:mask offset:0 atIndex:1];
                [encoder setBytes:&params length:sizeof(params) atIndex:2];
                hnw::metal::dispatch_1d(encoder, pipeline, plane_size);
                [encoder endEncoding];
            }
            id<MTLBuffer> result = mask;
            id<MTLBuffer> spare = scratch;
            encode_morphology(mask_pass, workspace, &result, &spare, params, open_ksize,
                              dilate_ksize, plane_size, kContext);
            [mask_pass commit];
            [mask_pass waitUntilCompleted];
            hnw::metal::throw_if_command_failed(mask_pass, kContext);

            std::memcpy(mask_host, result.contents, mask_bytes);
            workspace.finish_operation();
        } catch (...) {
            workspace.reset_after_error();
            throw;
        }
    }
}

template <typename T>
py::array_t<uint8_t>
star_mask_dog_metal_impl(const py::array_t<T, py::array::c_style | py::array::forcecast>& image,
                         const float sigma_small, const float sigma_large,
                         const float threshold_ratio, const int open_ksize,
                         const int dilate_ksize) {
    validate_common(image, threshold_ratio, open_ksize, dilate_ksize, kContext);
    const int height = static_cast<int>(image.shape(0));
    const int width = static_cast<int>(image.shape(1));
    const int channels = image.ndim() == 3 ? 3 : 1;
    const std::vector<float> small_kernel =
        hnw::camera::make_dog_gaussian_kernel(sigma_small, kContext);
    const std::vector<float> large_kernel =
        hnw::camera::make_dog_gaussian_kernel(sigma_large, kContext);
    py::array_t<uint8_t> output({height, width});
    {
        py::gil_scoped_release release;
        launch_star_mask_dog_metal(image.data(), output.mutable_data(), height, width, channels,
                                   small_kernel, large_kernel, threshold_ratio, open_ksize,
                                   dilate_ksize);
    }
    return output;
}

py::array star_mask_dog_metal_dispatch(const py::array& image, const float sigma_small,
                                       const float sigma_large, const float threshold_ratio,
                                       const int open_ksize, const int dilate_ksize) {
    if (py::isinstance<py::array_t<uint8_t>>(image)) {
        return star_mask_dog_metal_impl(
            image.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(),
            sigma_small, sigma_large, threshold_ratio, open_ksize, dilate_ksize);
    }
    if (py::isinstance<py::array_t<uint16_t>>(image)) {
        return star_mask_dog_metal_impl(
            image.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(),
            sigma_small, sigma_large, threshold_ratio, open_ksize, dilate_ksize);
    }
    throw std::invalid_argument(std::string(kContext) +
                                ": unsupported dtype; expected uint8 or uint16");
}

} // namespace

void bind_star_mask_dog_metal_ops(py::module_& m) {
    m.def("star_mask_dog_metal", &star_mask_dog_metal_dispatch, py::arg("image"),
          py::arg("sigma_small"), py::arg("sigma_large"), py::arg("threshold_ratio"),
          py::arg("open_ksize"), py::arg("dilate_ksize"),
          "Detect a star mask with the DoG Metal host-I/O backend.");
}
