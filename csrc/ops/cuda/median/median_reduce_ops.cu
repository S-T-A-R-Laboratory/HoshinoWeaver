#include "common/compat.h"
#include "common/cuda_host_io_workspace.cuh"
#include "common/host_parallel_copy.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kThreadsPerBlock = 256;

template <typename T, int SortSize>
__global__ void median_reduce_kernel(const T* stack, T* output, const int n_frames,
                                     const int64_t plane_size) {
    const int64_t pixel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pixel >= plane_size) {
        return;
    }

    T values[SortSize];
#pragma unroll
    for (int frame = 0; frame < SortSize; ++frame) {
        values[frame] = frame < n_frames ? stack[static_cast<int64_t>(frame) * plane_size + pixel]
                                         : static_cast<T>(-1);
    }
#pragma unroll
    for (int span = 2; span <= SortSize; span *= 2) {
#pragma unroll
        for (int stride = span / 2; stride > 0; stride /= 2) {
#pragma unroll
            for (int index = 0; index < SortSize; ++index) {
                const int partner = index ^ stride;
                if (partner > index) {
                    const T first = values[index];
                    const T second = values[partner];
                    const bool ascending = (index & span) == 0;
                    values[index] = ascending ? min(first, second) : max(first, second);
                    values[partner] = ascending ? max(first, second) : min(first, second);
                }
            }
        }
    }

    const int middle = n_frames / 2;
    T low = 0;
    T high = 0;
#pragma unroll
    for (int index = 0; index < SortSize; ++index) {
        if (index == middle) {
            high = values[index];
        }
        if (index + 1 == middle) {
            low = values[index];
        }
    }
    output[pixel] = (n_frames & 1) ? high : static_cast<T>((static_cast<uint32_t>(low) + high) / 2);
}

template <typename T, int SortSize>
void launch_median_kernel(const T* input, T* output, const int n_frames, const int64_t plane_size,
                          const cudaStream_t stream) {
    const int blocks = static_cast<int>((plane_size + kThreadsPerBlock - 1) / kThreadsPerBlock);
    median_reduce_kernel<T, SortSize>
        <<<blocks, kThreadsPerBlock, 0, stream>>>(input, output, n_frames, plane_size);
    hnw::cuda::throw_if_failed(cudaGetLastError(), "median_reduce_chunk_cuda kernel launch");
}

template <typename T, typename StageInput>
void launch_median_reduce_cuda(T* output_host, const int n_frames, const int64_t plane_size,
                               StageInput stage_input) {
    if (n_frames < 1 || n_frames > 128 || plane_size <= 0 ||
        plane_size > static_cast<int64_t>(std::numeric_limits<int>::max()) * kThreadsPerBlock ||
        static_cast<size_t>(plane_size) >
            std::numeric_limits<size_t>::max() / sizeof(T) / static_cast<size_t>(n_frames)) {
        throw std::invalid_argument("median_reduce_chunk_cuda: invalid stack dimensions");
    }
    auto workspace = hnw::cuda::acquire_host_io_workspace("median_reduce_chunk_cuda cudaGetDevice");
    try {
        const size_t input_bytes = static_cast<size_t>(n_frames) * plane_size * sizeof(T);
        const size_t output_bytes = static_cast<size_t>(plane_size) * sizeof(T);
        auto* input_device = static_cast<T*>(
            workspace.device_buffer(input_bytes, "median_reduce_chunk_cuda cudaMalloc(input)"));
        auto* output_device = static_cast<T*>(
            workspace.device_buffer(output_bytes, "median_reduce_chunk_cuda cudaMalloc(output)"));
        auto* stage = static_cast<T*>(
            workspace.pinned_buffer(input_bytes, "median_reduce_chunk_cuda cudaMallocHost"));
        const cudaStream_t stream = workspace.stream();

        stage_input(stage, input_bytes);
        hnw::cuda::throw_if_failed(
            cudaMemcpyAsync(input_device, stage, input_bytes, cudaMemcpyHostToDevice, stream),
            "median_reduce_chunk_cuda upload");
        if (n_frames <= 2) {
            launch_median_kernel<T, 2>(input_device, output_device, n_frames, plane_size, stream);
        } else if (n_frames <= 4) {
            launch_median_kernel<T, 4>(input_device, output_device, n_frames, plane_size, stream);
        } else if (n_frames <= 8) {
            launch_median_kernel<T, 8>(input_device, output_device, n_frames, plane_size, stream);
        } else if (n_frames <= 16) {
            launch_median_kernel<T, 16>(input_device, output_device, n_frames, plane_size, stream);
        } else if (n_frames <= 32) {
            launch_median_kernel<T, 32>(input_device, output_device, n_frames, plane_size, stream);
        } else if (n_frames <= 64) {
            launch_median_kernel<T, 64>(input_device, output_device, n_frames, plane_size, stream);
        } else {
            launch_median_kernel<T, 128>(input_device, output_device, n_frames, plane_size, stream);
        }
        // The same stream finishes reading stage before the download overwrites it.
        hnw::cuda::throw_if_failed(
            cudaMemcpyAsync(stage, output_device, output_bytes, cudaMemcpyDeviceToHost, stream),
            "median_reduce_chunk_cuda download");
        hnw::cuda::throw_if_failed(cudaStreamSynchronize(stream),
                                   "median_reduce_chunk_cuda cudaStreamSynchronize");
        hnw::parallel_copy(output_host, stage, output_bytes);
    } catch (...) {
        workspace.reset_after_error();
        throw;
    }
}

} // namespace

void launch_median_reduce_cuda_u8(const uint8_t* stack, uint8_t* output, const int n_frames,
                                  const int64_t plane_size) {
    launch_median_reduce_cuda(output, n_frames, plane_size, [stack](uint8_t* stage, size_t bytes) {
        hnw::parallel_copy(stage, stack, bytes);
    });
}

void launch_median_reduce_cuda_u16(const uint16_t* stack, uint16_t* output, const int n_frames,
                                   const int64_t plane_size) {
    launch_median_reduce_cuda(output, n_frames, plane_size, [stack](uint16_t* stage, size_t bytes) {
        hnw::parallel_copy(stage, stack, bytes);
    });
}

template <typename T>
void launch_median_reduce_frames_cuda(const T* const* frames, T* output, const int n_frames,
                                      const int64_t plane_size) {
    std::vector<const void*> source_ptrs(frames, frames + n_frames);
    launch_median_reduce_cuda(
        output, n_frames, plane_size, [&source_ptrs, n_frames, plane_size](T* stage, size_t) {
            const size_t frame_bytes = static_cast<size_t>(plane_size) * sizeof(T);
            hnw::parallel_copy_frames(stage, source_ptrs.data(), frame_bytes,
                                      static_cast<size_t>(n_frames));
        });
}

void launch_median_reduce_frames_cuda_u8(const uint8_t* const* frames, uint8_t* output,
                                         const int n_frames, const int64_t plane_size) {
    launch_median_reduce_frames_cuda(frames, output, n_frames, plane_size);
}

void launch_median_reduce_frames_cuda_u16(const uint16_t* const* frames, uint16_t* output,
                                          const int n_frames, const int64_t plane_size) {
    launch_median_reduce_frames_cuda(frames, output, n_frames, plane_size);
}
