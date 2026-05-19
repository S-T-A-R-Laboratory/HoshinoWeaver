#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

template <typename T>
struct CudaFreeDeleter {
    void operator()(T* ptr) const noexcept {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
};

template <typename T>
using device_ptr = std::unique_ptr<T, CudaFreeDeleter<T>>;

void throw_if_cuda_failed(const cudaError_t error, const char* context) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(context) + ": " + cudaGetErrorString(error));
    }
}

template <typename T>
__device__ inline T cast_output(float value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else if constexpr (std::is_same_v<T, unsigned char>) {
        value = fminf(fmaxf(value, 0.0f), 255.0f);
        return static_cast<unsigned char>(nearbyintf(value));
    } else {
        value = fminf(fmaxf(value, 0.0f), 65535.0f);
        return static_cast<unsigned short>(nearbyintf(value));
    }
}

template <typename T>
__global__ void camera_model_remap_fused_kernel(
    const T* image,
    T* out,
    const int src_height,
    const int src_width,
    const int channels,
    const int out_height,
    const int out_width,
    const float fx_src,
    const float fy_src,
    const float cx_src,
    const float cy_src,
    const float fx_dst,
    const float fy_dst,
    const float cx_dst,
    const float cy_dst,
    const float r00,
    const float r01,
    const float r02,
    const float r10,
    const float r11,
    const float r12,
    const float r20,
    const float r21,
    const float r22) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = out_height * out_width;
    if (idx >= total) {
        return;
    }

    const int row = idx / out_width;
    const int col = idx - row * out_width;
    const float x = (static_cast<float>(col) - cx_dst) / fx_dst;
    const float y = (static_cast<float>(row) - cy_dst) / fy_dst;

    const float proj_x = r00 * x + r01 * y + r02;
    const float proj_y = r10 * x + r11 * y + r12;
    const float proj_z = r20 * x + r21 * y + r22;
    const int out_base = idx * channels;

    if (!(proj_z > 0.0f)) {
        for (int c = 0; c < channels; ++c) {
            out[out_base + c] = static_cast<T>(0);
        }
        return;
    }

    const float inv_z = 1.0f / proj_z;
    const float src_x = fx_src * proj_x * inv_z + cx_src;
    const float src_y = fy_src * proj_y * inv_z + cy_src;

    if (!isfinite(src_x) || !isfinite(src_y)) {
        for (int c = 0; c < channels; ++c) {
            out[out_base + c] = static_cast<T>(0);
        }
        return;
    }

    const int x0 = static_cast<int>(floorf(src_x));
    const int y0 = static_cast<int>(floorf(src_y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float dx_raw = src_x - static_cast<float>(x0);
    const float dy_raw = src_y - static_cast<float>(y0);
    // Match OpenCV INTER_LINEAR's fixed-point interpolation table.
    const float dx = nearbyintf(dx_raw * 32.0f) * (1.0f / 32.0f);
    const float dy = nearbyintf(dy_raw * 32.0f) * (1.0f / 32.0f);
    const float w00 = (1.0f - dx) * (1.0f - dy);
    const float w01 = dx * (1.0f - dy);
    const float w10 = (1.0f - dx) * dy;
    const float w11 = dx * dy;

    for (int c = 0; c < channels; ++c) {
        float accum = 0.0f;
        if (x0 >= 0 && x0 < src_width && y0 >= 0 && y0 < src_height) {
            accum += w00 * static_cast<float>(image[(y0 * src_width + x0) * channels + c]);
        }
        if (x1 >= 0 && x1 < src_width && y0 >= 0 && y0 < src_height) {
            accum += w01 * static_cast<float>(image[(y0 * src_width + x1) * channels + c]);
        }
        if (x0 >= 0 && x0 < src_width && y1 >= 0 && y1 < src_height) {
            accum += w10 * static_cast<float>(image[(y1 * src_width + x0) * channels + c]);
        }
        if (x1 >= 0 && x1 < src_width && y1 >= 0 && y1 < src_height) {
            accum += w11 * static_cast<float>(image[(y1 * src_width + x1) * channels + c]);
        }
        out[out_base + c] = cast_output<T>(accum);
    }
}

template <typename T>
void launch_camera_model_remap_fused_impl(
    const T* image_host,
    T* out_host,
    const int src_height,
    const int src_width,
    const int channels,
    const int out_height,
    const int out_width,
    const float fx_src,
    const float fy_src,
    const float cx_src,
    const float cy_src,
    const float fx_dst,
    const float fy_dst,
    const float cx_dst,
    const float cy_dst,
    const float* rotation_dst_to_src) {
    const size_t src_total =
        static_cast<size_t>(src_height) * static_cast<size_t>(src_width) *
        static_cast<size_t>(channels);
    const size_t out_total =
        static_cast<size_t>(out_height) * static_cast<size_t>(out_width) *
        static_cast<size_t>(channels);
    const size_t src_bytes = src_total * sizeof(T);
    const size_t out_bytes = out_total * sizeof(T);

    T* image_device_raw = nullptr;
    T* out_device_raw = nullptr;
    throw_if_cuda_failed(cudaMalloc(&image_device_raw, src_bytes),
                         "camera_model_remap cudaMalloc(image)");
    throw_if_cuda_failed(cudaMalloc(&out_device_raw, out_bytes),
                         "camera_model_remap cudaMalloc(out)");
    device_ptr<T> image_device(image_device_raw);
    device_ptr<T> out_device(out_device_raw);

    throw_if_cuda_failed(cudaMemcpy(image_device.get(), image_host, src_bytes,
                                    cudaMemcpyHostToDevice),
                         "camera_model_remap cudaMemcpy(image)");

    constexpr int threads_per_block = 256;
    const int total_pixels = out_height * out_width;
    const int blocks = (total_pixels + threads_per_block - 1) / threads_per_block;

    camera_model_remap_fused_kernel<<<blocks, threads_per_block>>>(
        image_device.get(),
        out_device.get(),
        src_height,
        src_width,
        channels,
        out_height,
        out_width,
        fx_src,
        fy_src,
        cx_src,
        cy_src,
        fx_dst,
        fy_dst,
        cx_dst,
        cy_dst,
        rotation_dst_to_src[0],
        rotation_dst_to_src[1],
        rotation_dst_to_src[2],
        rotation_dst_to_src[3],
        rotation_dst_to_src[4],
        rotation_dst_to_src[5],
        rotation_dst_to_src[6],
        rotation_dst_to_src[7],
        rotation_dst_to_src[8]);
    throw_if_cuda_failed(cudaGetLastError(),
                         "camera_model_remap kernel launch");
    throw_if_cuda_failed(cudaMemcpy(out_host, out_device.get(), out_bytes,
                                    cudaMemcpyDeviceToHost),
                         "camera_model_remap cudaMemcpy(out)");
}

}  // namespace

void launch_camera_model_remap_fused_u8(
    const unsigned char* image_host,
    unsigned char* out_host,
    int src_height,
    int src_width,
    int channels,
    int out_height,
    int out_width,
    float fx_src,
    float fy_src,
    float cx_src,
    float cy_src,
    float fx_dst,
    float fy_dst,
    float cx_dst,
    float cy_dst,
    const float* rotation_dst_to_src) {
    launch_camera_model_remap_fused_impl<unsigned char>(
        image_host, out_host, src_height, src_width, channels, out_height,
        out_width, fx_src, fy_src, cx_src, cy_src, fx_dst, fy_dst, cx_dst,
        cy_dst, rotation_dst_to_src);
}

void launch_camera_model_remap_fused_u16(
    const unsigned short* image_host,
    unsigned short* out_host,
    int src_height,
    int src_width,
    int channels,
    int out_height,
    int out_width,
    float fx_src,
    float fy_src,
    float cx_src,
    float cy_src,
    float fx_dst,
    float fy_dst,
    float cx_dst,
    float cy_dst,
    const float* rotation_dst_to_src) {
    launch_camera_model_remap_fused_impl<unsigned short>(
        image_host, out_host, src_height, src_width, channels, out_height,
        out_width, fx_src, fy_src, cx_src, cy_src, fx_dst, fy_dst, cx_dst,
        cy_dst, rotation_dst_to_src);
}

void launch_camera_model_remap_fused_f32(
    const float* image_host,
    float* out_host,
    int src_height,
    int src_width,
    int channels,
    int out_height,
    int out_width,
    float fx_src,
    float fy_src,
    float cx_src,
    float cy_src,
    float fx_dst,
    float fy_dst,
    float cx_dst,
    float cy_dst,
    const float* rotation_dst_to_src) {
    launch_camera_model_remap_fused_impl<float>(
        image_host, out_host, src_height, src_width, channels, out_height,
        out_width, fx_src, fy_src, cx_src, cy_src, fx_dst, fy_dst, cx_dst,
        cy_dst, rotation_dst_to_src);
}
