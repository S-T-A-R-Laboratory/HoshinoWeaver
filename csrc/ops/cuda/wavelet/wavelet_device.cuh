#pragma once

// Shared device-only helpers for CUDA wavelet and star-detection kernels.
#include "common/cuda_host_io_workspace.cuh"
#include "common/cuda_runtime_utils.cuh"
#include "common/wavelet_geometry.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int DB8_FILTER_LEN = hnw::wavelet::kDb8FilterLength;
constexpr int DB8_DWT_OFFSET = -14;
constexpr int DB8_IDWT_OFFSET = 14;

__constant__ double DB8_DEC_LO[DB8_FILTER_LEN] = {
    -0.00011747678412476953, 0.0006754494064505693, -0.00039174037337694705, -0.004870352993451574,
    0.008746094047405777,    0.013981027917398282,  -0.044088253930794755,   -0.017369301001807547,
    0.12874742662047847,     0.0004724845739132828, -0.2840155429615469,     -0.015829105256349306,
    0.5853546836542067,      0.6756307362972898,    0.31287159091429995,     0.05441584224310401,
};

__constant__ double DB8_DEC_HI[DB8_FILTER_LEN] = {
    -0.05441584224310401, 0.31287159091429995,     -0.6756307362972898,    0.5853546836542067,
    0.015829105256349306, -0.2840155429615469,     -0.0004724845739132828, 0.12874742662047847,
    0.017369301001807547, -0.044088253930794755,   -0.013981027917398282,  0.008746094047405777,
    0.004870352993451574, -0.00039174037337694705, -0.0006754494064505693, -0.00011747678412476953,
};

__constant__ double DB8_REC_LO[DB8_FILTER_LEN] = {
    0.05441584224310401,   0.31287159091429995,     0.6756307362972898,    0.5853546836542067,
    -0.015829105256349306, -0.2840155429615469,     0.0004724845739132828, 0.12874742662047847,
    -0.017369301001807547, -0.044088253930794755,   0.013981027917398282,  0.008746094047405777,
    -0.004870352993451574, -0.00039174037337694705, 0.0006754494064505693, -0.00011747678412476953,
};

__constant__ double DB8_REC_HI[DB8_FILTER_LEN] = {
    -0.00011747678412476953, -0.0006754494064505693, -0.00039174037337694705, 0.004870352993451574,
    0.008746094047405777,    -0.013981027917398282,  -0.044088253930794755,   0.017369301001807547,
    0.12874742662047847,     -0.0004724845739132828, -0.2840155429615469,     0.015829105256349306,
    0.5853546836542067,      -0.6756307362972898,    0.31287159091429995,     -0.05441584224310401,
};

void throw_if_cuda_failed(const cudaError_t error, const char* context) {
    hnw::cuda::throw_if_failed(error, context);
}

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), bytes_(other.bytes_), lease_(std::move(other.lease_)) {
        other.ptr_ = nullptr;
        other.bytes_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            bytes_ = other.bytes_;
            lease_ = std::move(other.lease_);
            other.ptr_ = nullptr;
            other.bytes_ = 0;
        }
        return *this;
    }

    ~DeviceBuffer() { reset(); }

    void allocate(const size_t count, const char* context,
                  hnw::cuda::HostIoWorkspaceSession* workspace = nullptr) {
        reset();
        bytes_ = count * sizeof(double);
        if (workspace != nullptr) {
            lease_ = workspace->device_lease(bytes_, context);
            ptr_ = static_cast<double*>(lease_.get());
        } else {
            throw_if_cuda_failed(cudaMalloc(&ptr_, bytes_), context);
            hnw::cuda::record_device_memory_acquire(bytes_);
        }
    }

    void reset() noexcept {
        if (ptr_ != nullptr) {
            if (lease_.get() != nullptr) {
                lease_.reset();
            } else {
                cudaFree(ptr_);
                hnw::cuda::record_device_memory_release(bytes_);
            }
            ptr_ = nullptr;
            bytes_ = 0;
        }
    }

    double* get() const { return ptr_; }

private:
    double* ptr_ = nullptr;
    size_t bytes_ = 0;
    hnw::cuda::HostIoDeviceLease lease_;
};

template <typename T> class DeviceTypedBuffer {
public:
    DeviceTypedBuffer() = default;
    DeviceTypedBuffer(const DeviceTypedBuffer&) = delete;
    DeviceTypedBuffer& operator=(const DeviceTypedBuffer&) = delete;

    DeviceTypedBuffer(DeviceTypedBuffer&& other) noexcept
        : ptr_(other.ptr_), bytes_(other.bytes_), lease_(std::move(other.lease_)) {
        other.ptr_ = nullptr;
        other.bytes_ = 0;
    }

    DeviceTypedBuffer& operator=(DeviceTypedBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            bytes_ = other.bytes_;
            lease_ = std::move(other.lease_);
            other.ptr_ = nullptr;
            other.bytes_ = 0;
        }
        return *this;
    }

    ~DeviceTypedBuffer() { reset(); }

    void allocate(const size_t count, const char* context,
                  hnw::cuda::HostIoWorkspaceSession* workspace = nullptr) {
        reset();
        bytes_ = count * sizeof(T);
        if (workspace != nullptr) {
            lease_ = workspace->device_lease(bytes_, context);
            ptr_ = static_cast<T*>(lease_.get());
        } else {
            throw_if_cuda_failed(cudaMalloc(&ptr_, bytes_), context);
            hnw::cuda::record_device_memory_acquire(bytes_);
        }
    }

    void reset() noexcept {
        if (ptr_ != nullptr) {
            if (lease_.get() != nullptr) {
                lease_.reset();
            } else {
                cudaFree(ptr_);
                hnw::cuda::record_device_memory_release(bytes_);
            }
            ptr_ = nullptr;
            bytes_ = 0;
        }
    }

    T* get() const { return ptr_; }

private:
    T* ptr_ = nullptr;
    size_t bytes_ = 0;
    hnw::cuda::HostIoDeviceLease lease_;
};

struct DeviceDetailLevel {
    int h = 0;
    int w = 0;
    DeviceBuffer cH;
    DeviceBuffer cV;
    DeviceBuffer cD;
};

struct DeviceImage {
    DeviceBuffer data;
    int h = 0;
    int w = 0;
    size_t size = 0;
};

int dwt_len(const int n) {
    return hnw::wavelet::dwt_length(n);
}

int idwt_len(const int n) {
    return hnw::wavelet::idwt_length(n);
}

__device__ inline int symmetric_index_device(int idx, const int n) {
    if (idx >= 0 && idx < n) {
        return idx;
    }
    if (n <= 1) {
        return 0;
    }
    const int period = 2 * n;
    idx %= period;
    if (idx < 0) {
        idx += period;
    }
    if (idx < n) {
        return idx;
    }
    return period - 1 - idx;
}

// row_hi == nullptr computes only the low band (the finest level keeps no detail).
__global__ void dwt_rows_kernel(const double* input, double* row_lo, double* row_hi, const int h,
                                const int w, const int out_w) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = h * out_w;
    if (idx >= total) {
        return;
    }
    const int y = idx / out_w;
    const int x = idx - y * out_w;
    const bool keep_detail = row_hi != nullptr;
    double lo = 0.0;
    double hi = 0.0;
    for (int j = 0; j < DB8_FILTER_LEN; ++j) {
        const int src_x = symmetric_index_device(2 * x + j + DB8_DWT_OFFSET, w);
        const double value = input[y * w + src_x];
        const int rev_j = DB8_FILTER_LEN - 1 - j;
        lo += DB8_DEC_LO[rev_j] * value;
        if (keep_detail) {
            hi += DB8_DEC_HI[rev_j] * value;
        }
    }
    row_lo[idx] = lo;
    if (keep_detail) {
        row_hi[idx] = hi;
    }
}

// Without row_hi (cH/cV/cD are then nullptr too) only the approximation is formed.
__global__ void dwt_cols_kernel(const double* row_lo, const double* row_hi, double* approx,
                                double* cH, double* cV, double* cD, const int h, const int out_h,
                                const int out_w) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = out_h * out_w;
    if (idx >= total) {
        return;
    }
    const int y = idx / out_w;
    const int x = idx - y * out_w;
    const bool keep_detail = row_hi != nullptr;
    double ll = 0.0;
    double hl = 0.0;
    double lh = 0.0;
    double hh = 0.0;
    for (int j = 0; j < DB8_FILTER_LEN; ++j) {
        const int src_y = symmetric_index_device(2 * y + j + DB8_DWT_OFFSET, h);
        const int rev_j = DB8_FILTER_LEN - 1 - j;
        const double row_lo_value = row_lo[src_y * out_w + x];
        ll += DB8_DEC_LO[rev_j] * row_lo_value;
        if (keep_detail) {
            const double row_hi_value = row_hi[src_y * out_w + x];
            hl += DB8_DEC_HI[rev_j] * row_lo_value;
            lh += DB8_DEC_LO[rev_j] * row_hi_value;
            hh += DB8_DEC_HI[rev_j] * row_hi_value;
        }
    }
    approx[idx] = ll;
    if (keep_detail) {
        cH[idx] = hl;
        cV[idx] = lh;
        cD[idx] = hh;
    }
}

// Only taps with an even upsampled position contribute, and their parity
// equals the output parity: j = (y & 1) + 2 * m, visited in increasing order.
// zero_detail (finest level) leaves col_hi == nullptr: that band is all zero.
// The explicit intrinsics pin the rounding nvcc's contraction chose for the
// original expressions, so moving code across branches cannot change results.
__global__ void idwt_cols_kernel(const double* approx, const double* cH_arr, const double* cV_arr,
                                 const double* cD_arr, double* col_lo, double* col_hi,
                                 const int approx_stride, const int h, const int w, const int out_h,
                                 const bool zero_detail) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = out_h * w;
    if (idx >= total) {
        return;
    }
    const int y = idx / w;
    const int x = idx - y * w;
    double lo = 0.0;
    double hi = 0.0;
    for (int j = y & 1; j < DB8_FILTER_LEN; j += 2) {
        const int t = y + DB8_IDWT_OFFSET - j;
        const int src_y = symmetric_index_device(t / 2, h);
        const int approx_offset = src_y * approx_stride + x;
        const int detail_offset = src_y * w + x;
        const double cA = approx[approx_offset];
        const double cH = zero_detail ? 0.0 : cH_arr[detail_offset];
        lo = __dadd_rn(lo, __fma_rn(DB8_REC_LO[j], cA, __dmul_rn(DB8_REC_HI[j], cH)));
        if (!zero_detail) {
            const double cV = cV_arr[detail_offset];
            const double cD = cD_arr[detail_offset];
            hi = __dadd_rn(hi, __fma_rn(DB8_REC_HI[j], cD, __dmul_rn(DB8_REC_LO[j], cV)));
        }
    }
    col_lo[idx] = lo;
    if (!zero_detail) {
        col_hi[idx] = hi;
    }
}

__global__ void idwt_rows_kernel(const double* col_lo, const double* col_hi, double* output,
                                 const int out_h, const int w, const int out_w) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = out_h * out_w;
    if (idx >= total) {
        return;
    }
    const int y = idx / out_w;
    const int x = idx - y * out_w;
    double value = 0.0;
    for (int j = x & 1; j < DB8_FILTER_LEN; j += 2) {
        const int t = x + DB8_IDWT_OFFSET - j;
        const int src_x = symmetric_index_device(t / 2, w);
        const int offset = y * w + src_x;
        const double hi_value = col_hi == nullptr ? 0.0 : col_hi[offset];
        value = __dadd_rn(
            value, __fma_rn(DB8_REC_LO[j], col_lo[offset], __dmul_rn(DB8_REC_HI[j], hi_value)));
    }
    output[idx] = value;
}

DeviceImage wavelet_dec_rec_device(DeviceBuffer current, int current_h, int current_w,
                                   const int level, const int threads,
                                   hnw::cuda::HostIoWorkspaceSession* workspace = nullptr,
                                   const cudaStream_t stream = nullptr) {
    size_t current_size = static_cast<size_t>(current_h) * static_cast<size_t>(current_w);
    std::vector<DeviceDetailLevel> details(static_cast<size_t>(level));

    for (int idx = 0; idx < level; ++idx) {
        const int out_h = dwt_len(current_h);
        const int out_w = dwt_len(current_w);
        const size_t row_size = static_cast<size_t>(current_h) * static_cast<size_t>(out_w);
        const size_t detail_size = static_cast<size_t>(out_h) * static_cast<size_t>(out_w);
        // The reconstruction zeroes the finest detail band, so level 0 keeps none.
        const bool keep_detail = idx != 0;
        DeviceBuffer row_lo;
        DeviceBuffer row_hi;
        DeviceBuffer approx;
        row_lo.allocate(row_size, "wavelet_dec_rec_cuda_core cudaMalloc row_lo", workspace);
        if (keep_detail) {
            row_hi.allocate(row_size, "wavelet_dec_rec_cuda_core cudaMalloc row_hi", workspace);
        }
        approx.allocate(detail_size, "wavelet_dec_rec_cuda_core cudaMalloc approx", workspace);
        DeviceDetailLevel& detail = details[static_cast<size_t>(idx)];
        detail.h = out_h;
        detail.w = out_w;
        if (keep_detail) {
            detail.cH.allocate(detail_size, "wavelet_dec_rec_cuda_core cudaMalloc cH", workspace);
            detail.cV.allocate(detail_size, "wavelet_dec_rec_cuda_core cudaMalloc cV", workspace);
            detail.cD.allocate(detail_size, "wavelet_dec_rec_cuda_core cudaMalloc cD", workspace);
        }

        const int row_blocks = (static_cast<int>(row_size) + threads - 1) / threads;
        dwt_rows_kernel<<<row_blocks, threads, 0, stream>>>(
            current.get(), row_lo.get(), row_hi.get(), current_h, current_w, out_w);
        throw_if_cuda_failed(cudaGetLastError(), "wavelet_dec_rec_cuda_core dwt rows launch");
        const int col_blocks = (static_cast<int>(detail_size) + threads - 1) / threads;
        dwt_cols_kernel<<<col_blocks, threads, 0, stream>>>(
            row_lo.get(), row_hi.get(), approx.get(), detail.cH.get(), detail.cV.get(),
            detail.cD.get(), current_h, out_h, out_w);
        throw_if_cuda_failed(cudaGetLastError(), "wavelet_dec_rec_cuda_core dwt cols launch");

        current = std::move(approx);
        current_h = out_h;
        current_w = out_w;
        current_size = detail_size;
    }

    throw_if_cuda_failed(cudaMemsetAsync(current.get(), 0, current_size * sizeof(double), stream),
                         "wavelet_dec_rec_cuda_core cudaMemset approx");
    for (int idx = level - 1; idx >= 0; --idx) {
        DeviceDetailLevel& detail = details[static_cast<size_t>(idx)];
        const int out_h = idwt_len(detail.h);
        const int out_w = idwt_len(detail.w);
        const size_t col_size = static_cast<size_t>(out_h) * static_cast<size_t>(detail.w);
        const size_t out_size = static_cast<size_t>(out_h) * static_cast<size_t>(out_w);
        const bool zero_detail = idx == 0;
        DeviceBuffer col_lo;
        DeviceBuffer col_hi;
        DeviceBuffer output;
        col_lo.allocate(col_size, "wavelet_dec_rec_cuda_core cudaMalloc col_lo", workspace);
        if (!zero_detail) {
            col_hi.allocate(col_size, "wavelet_dec_rec_cuda_core cudaMalloc col_hi", workspace);
        }
        output.allocate(out_size, "wavelet_dec_rec_cuda_core cudaMalloc output", workspace);

        const int col_blocks = (static_cast<int>(col_size) + threads - 1) / threads;
        idwt_cols_kernel<<<col_blocks, threads, 0, stream>>>(
            current.get(), detail.cH.get(), detail.cV.get(), detail.cD.get(), col_lo.get(),
            col_hi.get(), current_w, detail.h, detail.w, out_h, zero_detail);
        throw_if_cuda_failed(cudaGetLastError(), "wavelet_dec_rec_cuda_core idwt cols launch");
        const int row_blocks = (static_cast<int>(out_size) + threads - 1) / threads;
        idwt_rows_kernel<<<row_blocks, threads, 0, stream>>>(col_lo.get(), col_hi.get(),
                                                             output.get(), out_h, detail.w, out_w);
        throw_if_cuda_failed(cudaGetLastError(), "wavelet_dec_rec_cuda_core idwt rows launch");

        detail.cH.reset();
        detail.cV.reset();
        detail.cD.reset();
        current = std::move(output);
        current_h = out_h;
        current_w = out_w;
        current_size = static_cast<size_t>(current_h) * static_cast<size_t>(current_w);
    }

    DeviceImage result;
    result.data = std::move(current);
    result.h = current_h;
    result.w = current_w;
    result.size = current_size;
    return result;
}

} // namespace
