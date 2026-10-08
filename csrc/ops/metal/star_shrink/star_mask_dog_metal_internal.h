#pragma once

// Objective-C++ only: include from a .mm, never a .cpp.
//
// Shared by the standalone DoG mask op and the fused DoG-shrink op. Both encode
// the same detection stage, so these four helpers would otherwise be duplicated
// across the two translation units.

#include "common/metal_dispatch.h"
#include "common/metal_host_io_workspace.h"
#include "star_shrink_metal_params.h"

#include <pybind11/numpy.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace hnw::metal::star_shrink {

inline void validate_common(const py::array& image, const float threshold_ratio,
                            const int open_ksize, const int dilate_ksize, const char* context) {
    const std::string prefix(context);
    if (image.ndim() != 2 && image.ndim() != 3) {
        throw std::invalid_argument(prefix + ": image must have shape (H, W) or (H, W, C)");
    }
    if (image.ndim() == 3 && image.shape(2) != 3) {
        throw std::invalid_argument(prefix + ": 3D image must have exactly 3 channels");
    }
    if (image.shape(0) <= 0 || image.shape(1) <= 0) {
        throw std::invalid_argument(prefix + ": image height and width must be positive");
    }
    if (image.shape(0) > std::numeric_limits<int>::max() ||
        image.shape(1) > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(prefix + ": image is too large");
    }
    const uint64_t channels = image.ndim() == 3 ? 3 : 1;
    const uint64_t total = static_cast<uint64_t>(image.shape(0)) * image.shape(1) * channels;
    if (total > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument(prefix + ": image is too large");
    }
    if (!std::isfinite(threshold_ratio)) {
        throw std::invalid_argument(prefix + ": threshold_ratio must be finite");
    }
    if (open_ksize < 0 || (open_ksize > 0 && open_ksize % 2 == 0)) {
        throw std::invalid_argument(prefix + ": open_ksize must be zero or a positive odd value");
    }
    if (dilate_ksize < 0 || (dilate_ksize > 0 && dilate_ksize % 2 == 0)) {
        throw std::invalid_argument(prefix + ": dilate_ksize must be zero or a positive odd value");
    }
}

inline void encode_gaussian(id<MTLCommandBuffer> command_buffer,
                            hnw::metal::HostIOWorkspace& workspace, id<MTLBuffer> input,
                            id<MTLBuffer> tmp, id<MTLBuffer> output, id<MTLBuffer> weights,
                            MaskParams params, const int radius, const uint32_t plane_size,
                            const char* context) {
    params.ksize = static_cast<uint32_t>(radius);
    {
        id<MTLComputePipelineState> pipeline = workspace.pipeline("star_mask_gaussian_horizontal");
        id<MTLComputeCommandEncoder> encoder =
            hnw::metal::begin_encoder(command_buffer, pipeline, context);
        [encoder setBuffer:input offset:0 atIndex:0];
        [encoder setBuffer:tmp offset:0 atIndex:1];
        [encoder setBuffer:weights offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        hnw::metal::dispatch_1d(encoder, pipeline, plane_size);
        [encoder endEncoding];
    }
    {
        id<MTLComputePipelineState> pipeline = workspace.pipeline("star_mask_gaussian_vertical");
        id<MTLComputeCommandEncoder> encoder =
            hnw::metal::begin_encoder(command_buffer, pipeline, context);
        [encoder setBuffer:tmp offset:0 atIndex:0];
        [encoder setBuffer:output offset:0 atIndex:1];
        [encoder setBuffer:weights offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        hnw::metal::dispatch_1d(encoder, pipeline, plane_size);
        [encoder endEncoding];
    }
}

inline void encode_morphology(id<MTLCommandBuffer> command_buffer,
                              hnw::metal::HostIOWorkspace& workspace, id<MTLBuffer>* mask,
                              id<MTLBuffer>* scratch, MaskParams params, const int open_ksize,
                              const int dilate_ksize, const uint32_t plane_size,
                              const char* context) {
    const auto encode = [&](const char* function_name, const int ksize) {
        params.ksize = static_cast<uint32_t>(ksize);
        id<MTLComputePipelineState> pipeline = workspace.pipeline(function_name);
        id<MTLComputeCommandEncoder> encoder =
            hnw::metal::begin_encoder(command_buffer, pipeline, context);
        [encoder setBuffer:*mask offset:0 atIndex:0];
        [encoder setBuffer:*scratch offset:0 atIndex:1];
        [encoder setBytes:&params length:sizeof(params) atIndex:2];
        hnw::metal::dispatch_1d(encoder, pipeline, plane_size);
        [encoder endEncoding];
        std::swap(*mask, *scratch);
    };
    if (open_ksize > 0) {
        encode("star_mask_erode_cross", open_ksize);
        encode("star_mask_dilate_cross", open_ksize);
    }
    if (dilate_ksize > 0) {
        encode("star_mask_dilate_cross", dilate_ksize);
    }
}

// The DoG mean/stddev reduction stays on the host in double precision: MSL has no
// fp64, and unified memory makes the buffer readable without a copy.
inline float compute_threshold(const float* dog, const uint32_t plane_size,
                               const float threshold_ratio) {
    double sum = 0.0;
    double square_sum = 0.0;
    for (uint32_t index = 0; index < plane_size; ++index) {
        const double value = static_cast<double>(dog[index]);
        sum += value;
        square_sum += value * value;
    }
    const double count = static_cast<double>(plane_size);
    const double mean = sum / count;
    const double variance = std::max(0.0, square_sum / count - mean * mean);
    return static_cast<float>(std::sqrt(variance) * static_cast<double>(threshold_ratio));
}

} // namespace hnw::metal::star_shrink
