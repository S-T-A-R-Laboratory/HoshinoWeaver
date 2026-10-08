#include "star_shrink_dog_process_ops.h"

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
using hnw::metal::star_shrink::parse_shrink_shape;
using hnw::metal::star_shrink::ShrinkParams;
using hnw::metal::star_shrink::validate_common;
using hnw::metal::star_shrink::validate_shrink_params;

constexpr const char* kFusedContext = "star_shrink_dog_process_metal";

template <typename T>
void launch_star_shrink_dog_process_metal(const T* image_host, T* out_host, const int height,
                                          const int width, const int channels,
                                          const std::vector<float>& small_kernel,
                                          const std::vector<float>& large_kernel,
                                          const float threshold_ratio, const int open_ksize,
                                          const int dilate_ksize, const int shrink_ksize,
                                          const int shrink_shape, const int shrink_times,
                                          const float shrink_ratio, const int deringing_ksize) {
    @autoreleasepool {
        auto& workspace = hnw::metal::HostIOWorkspace::current();
        workspace.begin_operation("star_shrink_dog_process");
        try {
            const uint32_t plane_size =
                static_cast<uint32_t>(static_cast<uint64_t>(height) * static_cast<uint64_t>(width));
            const uint32_t total = static_cast<uint32_t>(static_cast<uint64_t>(plane_size) *
                                                         static_cast<uint64_t>(channels));
            const size_t image_bytes = static_cast<size_t>(total) * sizeof(T);
            const size_t plane_float_bytes = static_cast<size_t>(plane_size) * sizeof(float);
            const size_t total_float_bytes = static_cast<size_t>(total) * sizeof(float);
            const size_t mask_bytes = static_cast<size_t>(plane_size);
            const size_t small_weight_bytes = small_kernel.size() * sizeof(float);
            const size_t large_weight_bytes = large_kernel.size() * sizeof(float);
            const char* suffix = std::is_same_v<T, uint8_t> ? "u8" : "u16";

            id<MTLBuffer> image = workspace.buffer(image_bytes, "star_shrink_dog_metal image");
            id<MTLBuffer> output = workspace.buffer(image_bytes, "star_shrink_dog_metal output");
            id<MTLBuffer> gray = workspace.buffer(plane_float_bytes, "star_shrink_dog_metal gray");
            id<MTLBuffer> tmp = workspace.buffer(plane_float_bytes, "star_shrink_dog_metal tmp");
            id<MTLBuffer> blur_small =
                workspace.buffer(plane_float_bytes, "star_shrink_dog_metal blur_small");
            id<MTLBuffer> blur_large =
                workspace.buffer(plane_float_bytes, "star_shrink_dog_metal blur_large");
            id<MTLBuffer> dog = workspace.buffer(plane_float_bytes, "star_shrink_dog_metal dog");
            id<MTLBuffer> small_weights =
                workspace.buffer(small_weight_bytes, "star_shrink_dog_metal small_weights");
            id<MTLBuffer> large_weights =
                workspace.buffer(large_weight_bytes, "star_shrink_dog_metal large_weights");
            id<MTLBuffer> mask = workspace.buffer(mask_bytes, "star_shrink_dog_metal mask");
            id<MTLBuffer> scratch = workspace.buffer(mask_bytes, "star_shrink_dog_metal scratch");
            // Detection's plane buffers are dead once the threshold kernel has
            // consumed dog, and the shrink stage needs exactly four, so alias
            // instead of allocating: 26MP saves about 415 MB on a pipeline that
            // is bandwidth-bound. Safe because separate compute encoders in one
            // command buffer run in order, so threshold precedes bgr_to_lab.
            id<MTLBuffer> luma = gray;
            id<MTLBuffer> luma_tmp = tmp;
            id<MTLBuffer> lab_a = blur_small;
            id<MTLBuffer> lab_b = blur_large;
            id<MTLBuffer> shrunk =
                workspace.buffer(total_float_bytes, "star_shrink_dog_metal shrunk");
            id<MTLBuffer> box_tmp =
                workspace.buffer(total_float_bytes, "star_shrink_dog_metal box_tmp");
            id<MTLBuffer> box_blurred =
                workspace.buffer(total_float_bytes, "star_shrink_dog_metal box_blurred");

            std::memcpy(image.contents, image_host, image_bytes);
            std::memcpy(small_weights.contents, small_kernel.data(), small_weight_bytes);
            std::memcpy(large_weights.contents, large_kernel.data(), large_weight_bytes);

            MaskParams mask_params{
                static_cast<uint32_t>(height),
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(channels),
                0,
                0.0f,
            };
            const ShrinkParams shrink_params{
                static_cast<uint32_t>(height),
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(channels),
                static_cast<uint32_t>(shrink_ksize),
                static_cast<uint32_t>(shrink_shape),
                static_cast<uint32_t>(deringing_ksize),
                shrink_ratio,
            };

            // Detection stops at the DoG plane: the host derives the threshold in
            // double, so this is the one sync the fusion cannot remove.
            id<MTLCommandBuffer> dog_pass =
                hnw::metal::new_command_buffer(workspace.command_queue(), kFusedContext);
            {
                const std::string name = std::string("star_mask_gray_") + suffix;
                id<MTLComputePipelineState> pipeline = workspace.pipeline(name.c_str());
                id<MTLComputeCommandEncoder> encoder =
                    hnw::metal::begin_encoder(dog_pass, pipeline, kFusedContext);
                [encoder setBuffer:image offset:0 atIndex:0];
                [encoder setBuffer:gray offset:0 atIndex:1];
                [encoder setBytes:&mask_params length:sizeof(mask_params) atIndex:2];
                hnw::metal::dispatch_1d(encoder, pipeline, plane_size);
                [encoder endEncoding];
            }
            encode_gaussian(dog_pass, workspace, gray, tmp, blur_small, small_weights, mask_params,
                            static_cast<int>(small_kernel.size() / 2), plane_size, kFusedContext);
            encode_gaussian(dog_pass, workspace, gray, tmp, blur_large, large_weights, mask_params,
                            static_cast<int>(large_kernel.size() / 2), plane_size, kFusedContext);
            {
                id<MTLComputePipelineState> pipeline = workspace.pipeline("star_mask_dog_diff");
                id<MTLComputeCommandEncoder> encoder =
                    hnw::metal::begin_encoder(dog_pass, pipeline, kFusedContext);
                [encoder setBuffer:blur_small offset:0 atIndex:0];
                [encoder setBuffer:blur_large offset:0 atIndex:1];
                [encoder setBuffer:dog offset:0 atIndex:2];
                [encoder setBytes:&mask_params length:sizeof(mask_params) atIndex:3];
                hnw::metal::dispatch_1d(encoder, pipeline, plane_size);
                [encoder endEncoding];
            }
            [dog_pass commit];
            [dog_pass waitUntilCompleted];
            hnw::metal::throw_if_command_failed(dog_pass, kFusedContext);

            mask_params.threshold = compute_threshold(static_cast<const float*>(dog.contents),
                                                      plane_size, threshold_ratio);

            // Everything after the threshold stays on the GPU: the mask is never
            // read back, which is the whole point of fusing the two ops.
            id<MTLCommandBuffer> shrink_pass =
                hnw::metal::new_command_buffer(workspace.command_queue(), kFusedContext);
            {
                id<MTLComputePipelineState> pipeline = workspace.pipeline("star_mask_threshold");
                id<MTLComputeCommandEncoder> encoder =
                    hnw::metal::begin_encoder(shrink_pass, pipeline, kFusedContext);
                [encoder setBuffer:dog offset:0 atIndex:0];
                [encoder setBuffer:mask offset:0 atIndex:1];
                [encoder setBytes:&mask_params length:sizeof(mask_params) atIndex:2];
                hnw::metal::dispatch_1d(encoder, pipeline, plane_size);
                [encoder endEncoding];
            }
            id<MTLBuffer> mask_result = mask;
            id<MTLBuffer> mask_spare = scratch;
            encode_morphology(shrink_pass, workspace, &mask_result, &mask_spare, mask_params,
                              open_ksize, dilate_ksize, plane_size, kFusedContext);
            {
                const std::string name = std::string("star_shrink_bgr_to_lab_") + suffix;
                id<MTLComputePipelineState> pipeline = workspace.pipeline(name.c_str());
                id<MTLComputeCommandEncoder> encoder =
                    hnw::metal::begin_encoder(shrink_pass, pipeline, kFusedContext);
                [encoder setBuffer:image offset:0 atIndex:0];
                [encoder setBuffer:luma offset:0 atIndex:1];
                [encoder setBuffer:lab_a offset:0 atIndex:2];
                [encoder setBuffer:lab_b offset:0 atIndex:3];
                [encoder setBytes:&shrink_params length:sizeof(shrink_params) atIndex:4];
                hnw::metal::dispatch_1d(encoder, pipeline, plane_size);
                [encoder endEncoding];
            }
            id<MTLBuffer> current = luma;
            id<MTLBuffer> next = luma_tmp;
            id<MTLComputePipelineState> erode_pipeline =
                workspace.pipeline("star_shrink_erode_luma");
            for (int iteration = 0; iteration < shrink_times; ++iteration) {
                id<MTLComputeCommandEncoder> encoder =
                    hnw::metal::begin_encoder(shrink_pass, erode_pipeline, kFusedContext);
                [encoder setBuffer:current offset:0 atIndex:0];
                [encoder setBuffer:next offset:0 atIndex:1];
                [encoder setBytes:&shrink_params length:sizeof(shrink_params) atIndex:2];
                hnw::metal::dispatch_1d(encoder, erode_pipeline, plane_size);
                [encoder endEncoding];
                std::swap(current, next);
            }
            {
                id<MTLComputePipelineState> pipeline = workspace.pipeline("star_shrink_lab_to_bgr");
                id<MTLComputeCommandEncoder> encoder =
                    hnw::metal::begin_encoder(shrink_pass, pipeline, kFusedContext);
                [encoder setBuffer:current offset:0 atIndex:0];
                [encoder setBuffer:lab_a offset:0 atIndex:1];
                [encoder setBuffer:lab_b offset:0 atIndex:2];
                [encoder setBuffer:shrunk offset:0 atIndex:3];
                [encoder setBytes:&shrink_params length:sizeof(shrink_params) atIndex:4];
                hnw::metal::dispatch_1d(encoder, pipeline, plane_size);
                [encoder endEncoding];
            }
            {
                const std::string name = std::string("star_shrink_horizontal_blur_") + suffix;
                id<MTLComputePipelineState> pipeline = workspace.pipeline(name.c_str());
                id<MTLComputeCommandEncoder> encoder =
                    hnw::metal::begin_encoder(shrink_pass, pipeline, kFusedContext);
                [encoder setBuffer:image offset:0 atIndex:0];
                [encoder setBuffer:box_tmp offset:0 atIndex:1];
                [encoder setBytes:&shrink_params length:sizeof(shrink_params) atIndex:2];
                hnw::metal::dispatch_1d(encoder, pipeline, total);
                [encoder endEncoding];
            }
            {
                id<MTLComputePipelineState> pipeline =
                    workspace.pipeline("star_shrink_vertical_blur");
                id<MTLComputeCommandEncoder> encoder =
                    hnw::metal::begin_encoder(shrink_pass, pipeline, kFusedContext);
                [encoder setBuffer:box_tmp offset:0 atIndex:0];
                [encoder setBuffer:box_blurred offset:0 atIndex:1];
                [encoder setBytes:&shrink_params length:sizeof(shrink_params) atIndex:2];
                hnw::metal::dispatch_1d(encoder, pipeline, total);
                [encoder endEncoding];
            }
            {
                const std::string name = std::string("star_shrink_final_mask_") + suffix;
                id<MTLComputePipelineState> pipeline = workspace.pipeline(name.c_str());
                id<MTLComputeCommandEncoder> encoder =
                    hnw::metal::begin_encoder(shrink_pass, pipeline, kFusedContext);
                [encoder setBuffer:image offset:0 atIndex:0];
                [encoder setBuffer:mask_result offset:0 atIndex:1];
                [encoder setBuffer:shrunk offset:0 atIndex:2];
                [encoder setBuffer:box_blurred offset:0 atIndex:3];
                [encoder setBuffer:output offset:0 atIndex:4];
                [encoder setBytes:&shrink_params length:sizeof(shrink_params) atIndex:5];
                hnw::metal::dispatch_1d(encoder, pipeline, total);
                [encoder endEncoding];
            }
            [shrink_pass commit];
            [shrink_pass waitUntilCompleted];
            hnw::metal::throw_if_command_failed(shrink_pass, kFusedContext);

            std::memcpy(out_host, output.contents, image_bytes);
            workspace.finish_operation();
        } catch (...) {
            workspace.reset_after_error();
            throw;
        }
    }
}

template <typename T>
py::array_t<T> star_shrink_dog_process_metal_impl(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& image, const float sigma_small,
    const float sigma_large, const float threshold_ratio, const int open_ksize,
    const int dilate_ksize, const int shrink_ksize, const std::string& shrink_shape,
    const int shrink_times, const float shrink_ratio, const int deringing_ksize) {
    validate_common(image, threshold_ratio, open_ksize, dilate_ksize, kFusedContext);
    validate_shrink_params(shrink_ksize, shrink_times, shrink_ratio, deringing_ksize,
                           kFusedContext);
    const int height = static_cast<int>(image.shape(0));
    const int width = static_cast<int>(image.shape(1));
    const int channels = image.ndim() == 3 ? 3 : 1;
    const int shape = parse_shrink_shape(shrink_shape, kFusedContext);
    const std::vector<float> small_kernel =
        hnw::camera::make_dog_gaussian_kernel(sigma_small, kFusedContext);
    const std::vector<float> large_kernel =
        hnw::camera::make_dog_gaussian_kernel(sigma_large, kFusedContext);
    py::array_t<T> output(image.request().shape);
    {
        py::gil_scoped_release release;
        launch_star_shrink_dog_process_metal(image.data(), output.mutable_data(), height, width,
                                             channels, small_kernel, large_kernel, threshold_ratio,
                                             open_ksize, dilate_ksize, shrink_ksize, shape,
                                             shrink_times, shrink_ratio, deringing_ksize);
    }
    return output;
}

py::array star_shrink_dog_process_metal_dispatch(const py::array& image, const float sigma_small,
                                                 const float sigma_large,
                                                 const float threshold_ratio, const int open_ksize,
                                                 const int dilate_ksize, const int shrink_ksize,
                                                 const std::string& shrink_shape,
                                                 const int shrink_times, const float shrink_ratio,
                                                 const int deringing_ksize) {
    if (py::isinstance<py::array_t<uint8_t>>(image)) {
        return star_shrink_dog_process_metal_impl(
            image.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(),
            sigma_small, sigma_large, threshold_ratio, open_ksize, dilate_ksize, shrink_ksize,
            shrink_shape, shrink_times, shrink_ratio, deringing_ksize);
    }
    if (py::isinstance<py::array_t<uint16_t>>(image)) {
        return star_shrink_dog_process_metal_impl(
            image.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(),
            sigma_small, sigma_large, threshold_ratio, open_ksize, dilate_ksize, shrink_ksize,
            shrink_shape, shrink_times, shrink_ratio, deringing_ksize);
    }
    throw std::invalid_argument(
        "star_shrink_dog_process_metal: unsupported dtype; expected uint8 or uint16");
}

} // namespace

void bind_star_shrink_dog_process_metal_ops(py::module_& m) {
    m.def("star_shrink_dog_process_metal", &star_shrink_dog_process_metal_dispatch,
          py::arg("image"), py::arg("sigma_small"), py::arg("sigma_large"),
          py::arg("threshold_ratio"), py::arg("open_ksize"), py::arg("dilate_ksize"),
          py::arg("shrink_ksize"), py::arg("shrink_shape"), py::arg("shrink_times"),
          py::arg("shrink_ratio"), py::arg("deringing_ksize"),
          "Run fused DoG detection and star shrinking with the Metal host-I/O backend.");
}
