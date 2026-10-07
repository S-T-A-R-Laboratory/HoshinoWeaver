#include "star_detect_fused_pixel_components_ops.h"

#include "../wavelet/wavelet_device.cuh"
#include "common/compat.h"
#include "common/cuda_error.h"
#include "common/cuda_host_io_workspace.cuh"
#include "common/cuda_host_staging.cuh"
#include "common/star_detect_capacity.h"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

template <typename Func>
decltype(auto) run_thrust_with_resource_translation(const char* context, Func&& func) {
    try {
        return func();
    } catch (const std::bad_alloc& exc) {
        throw hnw::CudaResourceExhaustedError(std::string(context) + ": " + exc.what());
    }
}

int next_power_of_two_int(const int value) {
    if (value <= 1) {
        return 1;
    }
    if (value > (std::numeric_limits<int>::max() / 2 + 1)) {
        throw std::runtime_error("star_detect_fused_pixel_components: hash table is too large");
    }
    unsigned int result = 1;
    const unsigned int target = static_cast<unsigned int>(value);
    while (result < target) {
        result <<= 1U;
    }
    return static_cast<int>(result);
}

__global__ void resize_linear_mask_kernel(const double* input, const uint8_t* mask, double* output,
                                          const int in_h, const int in_w, const int out_h,
                                          const int out_w) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = out_h * out_w;
    if (idx >= total) {
        return;
    }
    const int y = idx / out_w;
    const int x = idx - y * out_w;
    const double src_y =
        (static_cast<double>(y) + 0.5) * static_cast<double>(in_h) / static_cast<double>(out_h) -
        0.5;
    const double src_x =
        (static_cast<double>(x) + 0.5) * static_cast<double>(in_w) / static_cast<double>(out_w) -
        0.5;
    const int y0_raw = static_cast<int>(floor(src_y));
    const int x0_raw = static_cast<int>(floor(src_x));
    const double wy = src_y - static_cast<double>(y0_raw);
    const double wx = src_x - static_cast<double>(x0_raw);
    const int y0 = min(max(y0_raw, 0), in_h - 1);
    const int x0 = min(max(x0_raw, 0), in_w - 1);
    const int y1 = min(y0_raw + 1, in_h - 1);
    const int x1 = min(x0_raw + 1, in_w - 1);
    const double v00 = input[y0 * in_w + x0];
    const double v01 = input[y0 * in_w + x1];
    const double v10 = input[y1 * in_w + x0];
    const double v11 = input[y1 * in_w + x1];
    const double top = v00 + (v01 - v00) * wx;
    const double bottom = v10 + (v11 - v10) * wx;
    const double value = top + (bottom - top) * wy;
    output[idx] = mask[idx] == 0 ? 0.0 : value;
}

__global__ void resize_linear_kernel(const double* input, double* output, const int in_h,
                                     const int in_w, const int out_h, const int out_w) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = out_h * out_w;
    if (idx >= total) {
        return;
    }
    const int y = idx / out_w;
    const int x = idx - y * out_w;
    const double src_y =
        (static_cast<double>(y) + 0.5) * static_cast<double>(in_h) / static_cast<double>(out_h) -
        0.5;
    const double src_x =
        (static_cast<double>(x) + 0.5) * static_cast<double>(in_w) / static_cast<double>(out_w) -
        0.5;
    const int y0_raw = static_cast<int>(floor(src_y));
    const int x0_raw = static_cast<int>(floor(src_x));
    const double wy = src_y - static_cast<double>(y0_raw);
    const double wx = src_x - static_cast<double>(x0_raw);
    const int y0 = min(max(y0_raw, 0), in_h - 1);
    const int x0 = min(max(x0_raw, 0), in_w - 1);
    const int y1 = min(y0_raw + 1, in_h - 1);
    const int x1 = min(x0_raw + 1, in_w - 1);
    const double v00 = input[y0 * in_w + x0];
    const double v01 = input[y0 * in_w + x1];
    const double v10 = input[y1 * in_w + x0];
    const double v11 = input[y1 * in_w + x1];
    const double top = v00 + (v01 - v00) * wx;
    const double bottom = v10 + (v11 - v10) * wx;
    output[idx] = top + (bottom - top) * wy;
}

__device__ inline int reflect101_index(int idx, const int n) {
    if (n <= 1) {
        return 0;
    }
    while (idx < 0 || idx >= n) {
        if (idx < 0) {
            idx = -idx;
        } else {
            idx = 2 * n - idx - 2;
        }
    }
    return idx;
}

__global__ void gaussian_rows_kernel(const double* input, const double* kernel, double* output,
                                     const int height, const int width, const int ksize) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = height * width;
    if (idx >= total) {
        return;
    }
    const int y = idx / width;
    const int x = idx - y * width;
    const int radius = ksize / 2;
    double value = 0.0;
    for (int k = 0; k < ksize; ++k) {
        const int xx = reflect101_index(x + k - radius, width);
        value += input[y * width + xx] * kernel[k];
    }
    output[idx] = value;
}

__global__ void gaussian_cols_kernel(const double* input, const double* kernel, double* output,
                                     const int height, const int width, const int ksize) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = height * width;
    if (idx >= total) {
        return;
    }
    const int y = idx / width;
    const int x = idx - y * width;
    const int radius = ksize / 2;
    double value = 0.0;
    for (int k = 0; k < ksize; ++k) {
        const int yy = reflect101_index(y + k - radius, height);
        value += input[yy * width + x] * kernel[k];
    }
    output[idx] = value;
}

__global__ void normalize_kernel(const double* input, double* output, const double mean,
                                 const double range, const int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    output[idx] = (input[idx] - mean) / range;
}

__global__ void apply_mask_in_place_kernel(double* image, const uint8_t* mask, const int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total && mask[idx] == 0) {
        image[idx] = 0.0;
    }
}

__global__ void count_foreground_kernel(const uint8_t* bw, int* count, const int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const bool set = idx < total && bw[idx] != 0;
    const unsigned int votes = __ballot_sync(0xffffffffU, set);
    if ((threadIdx.x & 31) == 0 && votes != 0) {
        atomicAdd(count, __popc(votes));
    }
}

// Exact order statistics by radix selection over an order-preserving key: each
// pass histograms one 11-bit digit of the keys that match the digits already
// chosen, so only a tiny candidate set is gathered for the final nth_element.
constexpr int kSelectDigitBits = 11;
constexpr int kSelectBins = 1 << kSelectDigitBits;
constexpr int kSelectShifts[] = {53, 42, 31};

__device__ inline unsigned long long order_key(const double value) {
    const unsigned long long bits = static_cast<unsigned long long>(__double_as_longlong(value));
    return (bits >> 63) != 0 ? ~bits : bits | (1ULL << 63);
}

__global__ void masked_digit_histogram_kernel(const double* values, const uint8_t* mask,
                                              unsigned int* histogram, const int total,
                                              const int digit_shift, const int prefix_shift,
                                              const unsigned long long prefix) {
    __shared__ unsigned int local[kSelectBins];
    for (int bin = threadIdx.x; bin < kSelectBins; bin += blockDim.x) {
        local[bin] = 0;
    }
    __syncthreads();
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += gridDim.x * blockDim.x) {
        if (mask[idx] == 0) {
            continue;
        }
        const unsigned long long key = order_key(values[idx]);
        if (prefix_shift < 64 && (key >> prefix_shift) != prefix) {
            continue;
        }
        atomicAdd(&local[(key >> digit_shift) & (kSelectBins - 1)], 1U);
    }
    __syncthreads();
    for (int bin = threadIdx.x; bin < kSelectBins; bin += blockDim.x) {
        if (local[bin] != 0) {
            atomicAdd(&histogram[bin], local[bin]);
        }
    }
}

__global__ void gather_masked_prefix_kernel(const double* values, const uint8_t* mask,
                                            double* candidates, int* count, const int total,
                                            const int prefix_shift,
                                            const unsigned long long prefix) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += gridDim.x * blockDim.x) {
        if (mask[idx] != 0 && (order_key(values[idx]) >> prefix_shift) == prefix) {
            candidates[atomicAdd(count, 1)] = values[idx];
        }
    }
}

__device__ inline uint8_t threshold_pixel(const double value, const uint8_t mask,
                                          const double threshold) {
    return (mask != 0 && value > threshold) ? 255 : 0;
}

__global__ void erode_threshold_kernel(const double* image, const uint8_t* mask, uint8_t* eroded,
                                       const int height, const int width, const double threshold) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = height * width;
    if (idx >= total) {
        return;
    }
    const int y = idx / width;
    const int x = idx - y * width;
    uint8_t value = 255;
    for (int dy = -1; dy <= 1; ++dy) {
        const int yy = y + dy;
        if (yy < 0 || yy >= height) {
            continue;
        }
        for (int dx = -1; dx <= 1; ++dx) {
            const int xx = x + dx;
            if (xx < 0 || xx >= width) {
                continue;
            }
            const int offset = yy * width + xx;
            const uint8_t candidate = threshold_pixel(image[offset], mask[offset], threshold);
            value = min(value, candidate);
        }
    }
    eroded[idx] = value;
}

__global__ void dilate_kernel(const uint8_t* eroded, uint8_t* out, const int height,
                              const int width) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = height * width;
    if (idx >= total) {
        return;
    }
    const int y = idx / width;
    const int x = idx - y * width;
    uint8_t value = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        const int yy = y + dy;
        if (yy < 0 || yy >= height) {
            continue;
        }
        for (int dx = -1; dx <= 1; ++dx) {
            const int xx = x + dx;
            if (xx < 0 || xx >= width) {
                continue;
            }
            value = max(value, eroded[yy * width + xx]);
        }
    }
    out[idx] = value;
}

__global__ void init_component_labels_kernel(const uint8_t* bw, int* labels, const int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    labels[idx] = bw[idx] == 0 ? 0 : idx + 1;
}

__global__ void propagate_foreground_component_labels_kernel(const int* foreground_indices,
                                                             const int foreground_count,
                                                             const int* in_labels, int* out_labels,
                                                             int* changed, const int height,
                                                             const int width) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= foreground_count) {
        return;
    }
    const int idx = foreground_indices[item];
    const int current = in_labels[idx];
    const int y = idx / width;
    const int x = idx - y * width;
    int best = current;
    for (int dy = -1; dy <= 1; ++dy) {
        const int yy = y + dy;
        if (yy < 0 || yy >= height) {
            continue;
        }
        for (int dx = -1; dx <= 1; ++dx) {
            const int xx = x + dx;
            if (xx < 0 || xx >= width) {
                continue;
            }
            const int candidate = in_labels[yy * width + xx];
            if (candidate != 0 && candidate < best) {
                best = candidate;
            }
        }
    }
    out_labels[idx] = best;
    if (best != current) {
        atomicAdd(changed, 1);
    }
}

__global__ void compact_foreground_indices_kernel(const uint8_t* bw, int* indices, int* count,
                                                  const int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total || bw[idx] == 0) {
        return;
    }
    const int pos = atomicAdd(count, 1);
    indices[pos] = idx;
}

__device__ inline unsigned int component_hash(const unsigned int value) {
    unsigned int x = value;
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__device__ int find_component_slot(int* keys, int* overflow, const int capacity,
                                   const int root_label) {
    const int mask = capacity - 1;
    unsigned int pos =
        component_hash(static_cast<unsigned int>(root_label)) & static_cast<unsigned int>(mask);
    for (int probe = 0; probe < capacity; ++probe) {
        const int old = atomicCAS(&keys[pos], 0, root_label);
        if (old == 0 || old == root_label) {
            return static_cast<int>(pos);
        }
        pos = (pos + 1U) & static_cast<unsigned int>(mask);
    }
    atomicExch(overflow, 1);
    return -1;
}

__global__ void accumulate_component_stats_kernel(const int* foreground_indices,
                                                  const int foreground_count, const int* labels,
                                                  const double* image, int* keys, int* counts,
                                                  double* sum_x, double* sum_y,
                                                  double* sum_intensity, int* overflow,
                                                  const int width, const int hash_capacity) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= foreground_count) {
        return;
    }
    const int idx = foreground_indices[item];
    const int root = labels[idx];
    if (root == 0) {
        return;
    }
    const int slot = find_component_slot(keys, overflow, hash_capacity, root);
    if (slot < 0) {
        return;
    }

    const double x = static_cast<double>(idx % width);
    const double y = static_cast<double>(idx / width);
    atomicAdd(&counts[slot], 1);
    atomicAdd(&sum_x[slot], x);
    atomicAdd(&sum_y[slot], y);
    atomicAdd(&sum_intensity[slot], image[idx]);
}

__global__ void count_component_outputs_kernel(const int* keys, const int* counts, int* out_count,
                                               const int hash_capacity) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hash_capacity) {
        return;
    }
    if (keys[idx] != 0 && counts[idx] > 0) {
        atomicAdd(out_count, 1);
    }
}

__global__ void fill_component_outputs_kernel(const int* keys, const int* counts,
                                              const double* sum_x, const double* sum_y,
                                              const double* sum_intensity, double* positions,
                                              double* intensities, int* out_index,
                                              const int hash_capacity) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hash_capacity || keys[idx] == 0 || counts[idx] <= 0) {
        return;
    }

    const int out = atomicAdd(out_index, 1);
    const double count = static_cast<double>(counts[idx]);
    const double inv_count = 1.0 / count;
    const double cx = sum_x[idx] * inv_count;
    const double cy = sum_y[idx] * inv_count;

    positions[out * 2] = cx;
    positions[out * 2 + 1] = cy;
    intensities[out] = sum_intensity[idx] * inv_count;
}

struct SelectWorkspace {
    const double* values = nullptr;
    const uint8_t* mask = nullptr;
    int total = 0;
    unsigned int* histogram_device = nullptr;
    unsigned int* histogram_host = nullptr;
    int* count_device = nullptr;
    hnw::cuda::HostIoWorkspaceSession* workspace = nullptr;
    cudaStream_t stream = nullptr;
    int grid = 0;
    int threads = 0;
};

// Host copy of the digit histogram at digit_shift for keys matching prefix.
std::vector<unsigned int> masked_digit_histogram(const SelectWorkspace& select,
                                                 const int digit_shift, const int prefix_shift,
                                                 const unsigned long long prefix) {
    const size_t histogram_bytes = kSelectBins * sizeof(unsigned int);
    throw_if_cuda_failed(
        cudaMemsetAsync(select.histogram_device, 0, histogram_bytes, select.stream),
        "star_detect_fused_pixel_components cudaMemset select histogram");
    masked_digit_histogram_kernel<<<select.grid, select.threads, 0, select.stream>>>(
        select.values, select.mask, select.histogram_device, select.total, digit_shift,
        prefix_shift, prefix);
    throw_if_cuda_failed(cudaGetLastError(),
                         "star_detect_fused_pixel_components select histogram launch");
    throw_if_cuda_failed(cudaMemcpyAsync(select.histogram_host, select.histogram_device,
                                         histogram_bytes, cudaMemcpyDeviceToHost, select.stream),
                         "star_detect_fused_pixel_components cudaMemcpy select histogram");
    throw_if_cuda_failed(cudaStreamSynchronize(select.stream),
                         "star_detect_fused_pixel_components select histogram sync");
    return std::vector<unsigned int>(select.histogram_host, select.histogram_host + kSelectBins);
}

// Masked values sharing the 33-bit key prefix that contains sorted position
// `*rank` (0-based); on return `*rank` indexes into them.
std::vector<double> masked_rank_candidates(const SelectWorkspace& select,
                                           const std::vector<unsigned int>& top_histogram,
                                           uint64_t* rank) {
    std::vector<unsigned int> histogram = top_histogram;
    unsigned long long prefix = 0;
    int prefix_shift = 64;
    for (const int digit_shift : kSelectShifts) {
        if (digit_shift != kSelectShifts[0]) {
            histogram = masked_digit_histogram(select, digit_shift, prefix_shift, prefix);
        }
        int bin = 0;
        while (*rank >= histogram[bin]) {
            *rank -= histogram[bin];
            ++bin;
        }
        prefix = (prefix << kSelectDigitBits) | static_cast<unsigned long long>(bin);
        prefix_shift = digit_shift;
    }
    const int candidate_count = static_cast<int>(histogram[prefix & (kSelectBins - 1)]);

    DeviceBuffer candidates;
    candidates.allocate(static_cast<size_t>(candidate_count),
                        "star_detect_fused_pixel_components cudaMalloc select candidates",
                        select.workspace);
    throw_if_cuda_failed(cudaMemsetAsync(select.count_device, 0, sizeof(int), select.stream),
                         "star_detect_fused_pixel_components cudaMemset select count");
    gather_masked_prefix_kernel<<<select.grid, select.threads, 0, select.stream>>>(
        select.values, select.mask, candidates.get(), select.count_device, select.total,
        prefix_shift, prefix);
    throw_if_cuda_failed(cudaGetLastError(),
                         "star_detect_fused_pixel_components select gather launch");
    std::vector<double> host(static_cast<size_t>(candidate_count));
    throw_if_cuda_failed(cudaMemcpyAsync(host.data(), candidates.get(),
                                         host.size() * sizeof(double), cudaMemcpyDeviceToHost,
                                         select.stream),
                         "star_detect_fused_pixel_components cudaMemcpy select candidates");
    throw_if_cuda_failed(cudaStreamSynchronize(select.stream),
                         "star_detect_fused_pixel_components select gather sync");
    return host;
}

// np.percentile(values[mask], 99.5): the same order statistics the former
// full sort produced, selected without materializing the masked values.
double masked_percentile_995(const SelectWorkspace& select) {
    const std::vector<unsigned int> top_histogram =
        masked_digit_histogram(select, kSelectShifts[0], 64, 0);
    uint64_t count = 0;
    for (const unsigned int bin_count : top_histogram) {
        count += bin_count;
    }
    if (count == 0) {
        throw std::runtime_error("star_detect_fused_pixel_components: mask selects no pixels");
    }
    const double rank = 0.995 * static_cast<double>(count - 1);
    const uint64_t lower_index = static_cast<uint64_t>(floor(rank));
    const uint64_t upper_index = static_cast<uint64_t>(ceil(rank));

    uint64_t local_rank = lower_index;
    std::vector<double> candidates = masked_rank_candidates(select, top_histogram, &local_rank);
    const auto lower_it = candidates.begin() + static_cast<ptrdiff_t>(local_rank);
    std::nth_element(candidates.begin(), lower_it, candidates.end());
    const double lower = *lower_it;
    if (lower_index == upper_index) {
        return lower;
    }
    double upper = 0.0;
    if (local_rank + 1 < candidates.size()) {
        std::nth_element(lower_it + 1, lower_it + 1, candidates.end());
        upper = *(lower_it + 1);
    } else {
        uint64_t upper_rank = upper_index;
        std::vector<double> next = masked_rank_candidates(select, top_histogram, &upper_rank);
        const auto upper_it = next.begin() + static_cast<ptrdiff_t>(upper_rank);
        std::nth_element(next.begin(), upper_it, next.end());
        upper = *upper_it;
    }
    return lower + (upper - lower) * (rank - static_cast<double>(lower_index));
}

// to_gray_f64 on the device. OpenCV routes float BGR->GRAY to IPP, whose
// AVX2/AVX-512 code computes fma(r, cr, fma(b, cb, g * cg)); the even lanes of
// the 4-pixel block closing a row with 4..7 leftover pixels use
// fma(r, cr, fma(g, cg, b * cb)). The host verifies this against OpenCV.
template <typename T>
__global__ void source_gray_kernel(const T* source, const int height, const int width,
                                   const int channels, const double scale, double* gray) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= height * width) {
        return;
    }
    if (channels == 1) {
        gray[idx] = __ddiv_rn(static_cast<double>(source[idx]), scale);
        return;
    }
    const T* pixel = source + static_cast<size_t>(idx) * 3;
    const float b = pixel[0];
    const float g = pixel[1];
    const float r = pixel[2];
    const int col = idx % width;
    const int block = width / 8 * 8;
    const bool tail_even = width % 8 >= 4 && (col == block || col == block + 2);
    const float y = tail_even ? __fmaf_rn(r, 0.299f, __fmaf_rn(g, 0.587f, __fmul_rn(b, 0.114f)))
                              : __fmaf_rn(r, 0.299f, __fmaf_rn(b, 0.114f, __fmul_rn(g, 0.587f)));
    gray[idx] = __ddiv_rn(static_cast<double>(y), scale);
}

// Uploads `source` and writes its float64 gray into `gray`. A converted
// source is staged in `scratch` (at least 8 bytes per pixel).
void upload_source_gray(const StarDetectSource& source, const int height, const int width,
                        double* gray, void* scratch, unsigned char* staging,
                        hnw::cuda::StagingSlots* slots, const cudaStream_t stream) {
    const size_t total = static_cast<size_t>(height) * static_cast<size_t>(width);
    if (source.is_gray()) {
        hnw::cuda::staged_upload(gray, source.data, total * sizeof(double), staging, slots, stream);
        return;
    }
    const size_t bytes = total * static_cast<size_t>(source.channels * source.sample_bytes);
    hnw::cuda::staged_upload(scratch, source.data, bytes, staging, slots, stream);
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    if (source.sample_bytes == 1) {
        source_gray_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const uint8_t*>(scratch), height, width, source.channels, 255.0, gray);
    } else {
        source_gray_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const uint16_t*>(scratch), height, width, source.channels, 65535.0, gray);
    }
    throw_if_cuda_failed(cudaGetLastError(), "star_detect_fused_pixel_components gray launch");
}

} // namespace

void launch_star_detect_source_gray(const StarDetectSource& source, const int height,
                                    const int width, double* gray_host) {
    const size_t total = static_cast<size_t>(height) * static_cast<size_t>(width);
    auto workspace = hnw::cuda::acquire_host_io_workspace("star_detect_source_gray cudaGetDevice");
    try {
        const cudaStream_t stream = workspace.stream();
        auto* staging = static_cast<unsigned char*>(workspace.pinned_buffer(
            2 * hnw::cuda::kStagingSlotBytes, "star_detect_source_gray cudaMallocHost staging"));
        hnw::cuda::StagingSlots slots;
        void* scratch = workspace.device_buffer(total * sizeof(double),
                                                "star_detect_source_gray cudaMalloc source");
        auto* gray = static_cast<double*>(workspace.device_buffer(
            total * sizeof(double), "star_detect_source_gray cudaMalloc gray"));
        upload_source_gray(source, height, width, gray, scratch, staging, &slots, stream);
        throw_if_cuda_failed(cudaMemcpyAsync(gray_host, gray, total * sizeof(double),
                                             cudaMemcpyDeviceToHost, stream),
                             "star_detect_source_gray cudaMemcpy gray");
        throw_if_cuda_failed(cudaStreamSynchronize(stream),
                             "star_detect_source_gray cudaStreamSynchronize");
    } catch (...) {
        workspace.reset_after_error();
        throw;
    }
}

bool launch_star_detect_fused_pixel_components(
    const StarDetectSource& source, const uint8_t* external_mask_host,
    const double* gaussian_kernel_host, std::vector<double>* positions_xy_host,
    std::vector<double>* intensities_host, uint8_t* binary_mask_host, const int height,
    const int width, const int small_height, const int small_width, const int level,
    const int gaussian_ksize) {
    const int threads = 256;
    const int total = height * width;
    const int small_total = small_height * small_width;
    const size_t plane_size = static_cast<size_t>(total);
    const size_t small_size = static_cast<size_t>(small_total);

    auto workspace =
        hnw::cuda::acquire_host_io_workspace("star_detect_fused_pixel_components cudaGetDevice");
    try {
        const size_t mask_bytes = plane_size * sizeof(uint8_t);
        const size_t gaussian_kernel_bytes = static_cast<size_t>(gaussian_ksize) * sizeof(double);
        const cudaStream_t stream = workspace.stream();
        auto* scalar_doubles = static_cast<double*>(workspace.pinned_buffer(
            2 * sizeof(double),
            "star_detect_fused_pixel_components cudaMallocHost scalar doubles"));
        auto* scalar_int = static_cast<int*>(workspace.pinned_buffer(
            sizeof(int), "star_detect_fused_pixel_components cudaMallocHost scalar int"));
        auto* staging = static_cast<unsigned char*>(
            workspace.pinned_buffer(2 * hnw::cuda::kStagingSlotBytes,
                                    "star_detect_fused_pixel_components cudaMallocHost staging"));
        auto* histogram_host = static_cast<unsigned int*>(workspace.pinned_buffer(
            kSelectBins * sizeof(unsigned int),
            "star_detect_fused_pixel_components cudaMallocHost select histogram"));
        hnw::cuda::StagingSlots staging_slots;
        DeviceBuffer image;
        DeviceBuffer gaussian_kernel;
        DeviceBuffer blur_rows;
        DeviceBuffer small_blur;
        DeviceBuffer img_rec;
        DeviceTypedBuffer<uint8_t> mask;
        DeviceTypedBuffer<uint8_t> eroded;
        DeviceTypedBuffer<uint8_t> bw;
        DeviceTypedBuffer<int> count;
        DeviceTypedBuffer<unsigned int> select_histogram;
        DeviceTypedBuffer<int> foreground_indices;
        DeviceTypedBuffer<int> labels_a;
        DeviceTypedBuffer<int> labels_b;
        DeviceTypedBuffer<int> changed;
        DeviceTypedBuffer<int> keys;
        DeviceTypedBuffer<int> counts;
        DeviceTypedBuffer<int> overflow;
        DeviceBuffer sum_x;
        DeviceBuffer sum_y;
        DeviceBuffer sum_intensity;
        DeviceBuffer out_positions;
        DeviceBuffer out_intensities;

        positions_xy_host->clear();
        intensities_host->clear();

        image.allocate(plane_size, "star_detect_fused_pixel_components cudaMalloc image",
                       &workspace);
        gaussian_kernel.allocate(static_cast<size_t>(gaussian_ksize),
                                 "star_detect_fused_pixel_components cudaMalloc gaussian kernel",
                                 &workspace);
        blur_rows.allocate(plane_size, "star_detect_fused_pixel_components cudaMalloc blur rows",
                           &workspace);
        mask.allocate(plane_size, "star_detect_fused_pixel_components cudaMalloc mask", &workspace);
        count.allocate(1, "star_detect_fused_pixel_components cudaMalloc count", &workspace);

        // A converted source is staged in the blur-row buffer, which the
        // gaussian row pass overwrites only after the conversion.
        upload_source_gray(source, height, width, image.get(), blur_rows.get(), staging,
                           &staging_slots, stream);
        if (!source.is_gray()) {
            thrust::device_ptr<double> gray_ptr(image.get());
            const auto extrema =
                run_thrust_with_resource_translation("star_detect thrust minmax allocation", [&] {
                    return thrust::minmax_element(thrust::cuda::par.on(stream), gray_ptr,
                                                  gray_ptr + total);
                });
            throw_if_cuda_failed(cudaMemcpyAsync(scalar_doubles,
                                                 thrust::raw_pointer_cast(extrema.first),
                                                 sizeof(double), cudaMemcpyDeviceToHost, stream),
                                 "star_detect_fused_pixel_components cudaMemcpy gray min");
            throw_if_cuda_failed(cudaMemcpyAsync(scalar_doubles + 1,
                                                 thrust::raw_pointer_cast(extrema.second),
                                                 sizeof(double), cudaMemcpyDeviceToHost, stream),
                                 "star_detect_fused_pixel_components cudaMemcpy gray max");
            throw_if_cuda_failed(cudaStreamSynchronize(stream),
                                 "star_detect_fused_pixel_components gray extrema sync");
            if (scalar_doubles[0] == scalar_doubles[1]) {
                return false;
            }
        }
        throw_if_cuda_failed(cudaMemcpyAsync(gaussian_kernel.get(), gaussian_kernel_host,
                                             gaussian_kernel_bytes, cudaMemcpyHostToDevice, stream),
                             "star_detect_fused_pixel_components cudaMemcpy gaussian kernel");
        if (external_mask_host != nullptr) {
            hnw::cuda::staged_upload(mask.get(), external_mask_host, mask_bytes, staging,
                                     &staging_slots, stream);
        } else {
            throw_if_cuda_failed(cudaMemsetAsync(mask.get(), 1, mask_bytes, stream),
                                 "star_detect_fused_pixel_components cudaMemset full mask");
        }

        const int blocks = (total + threads - 1) / threads;
        const int small_blocks = (small_total + threads - 1) / threads;

        // The column pass writes the blur back into the image buffer, which
        // the row pass no longer needs.
        gaussian_rows_kernel<<<blocks, threads, 0, stream>>>(
            image.get(), gaussian_kernel.get(), blur_rows.get(), height, width, gaussian_ksize);
        throw_if_cuda_failed(cudaGetLastError(),
                             "star_detect_fused_pixel_components gaussian rows launch");
        gaussian_cols_kernel<<<blocks, threads, 0, stream>>>(
            blur_rows.get(), gaussian_kernel.get(), image.get(), height, width, gaussian_ksize);
        throw_if_cuda_failed(cudaGetLastError(),
                             "star_detect_fused_pixel_components gaussian cols launch");
        blur_rows.reset();
        gaussian_kernel.reset();

        thrust::device_ptr<double> blur_ptr(image.get());
        const double blur_sum =
            run_thrust_with_resource_translation("star_detect thrust reduce allocation", [&] {
                return thrust::reduce(thrust::cuda::par.on(stream), blur_ptr, blur_ptr + total,
                                      0.0);
            });
        const auto minmax =
            run_thrust_with_resource_translation("star_detect thrust minmax allocation", [&] {
                return thrust::minmax_element(thrust::cuda::par.on(stream), blur_ptr,
                                              blur_ptr + total);
            });
        throw_if_cuda_failed(cudaMemcpyAsync(scalar_doubles, thrust::raw_pointer_cast(minmax.first),
                                             sizeof(double), cudaMemcpyDeviceToHost, stream),
                             "star_detect_fused_pixel_components cudaMemcpy blur min");
        throw_if_cuda_failed(cudaMemcpyAsync(scalar_doubles + 1,
                                             thrust::raw_pointer_cast(minmax.second),
                                             sizeof(double), cudaMemcpyDeviceToHost, stream),
                             "star_detect_fused_pixel_components cudaMemcpy blur max");
        throw_if_cuda_failed(cudaStreamSynchronize(stream),
                             "star_detect_fused_pixel_components blur extrema sync");
        const double blur_min = scalar_doubles[0];
        const double blur_max = scalar_doubles[1];
        const double blur_range = blur_max - blur_min;
        if (!(blur_range > 0.0)) {
            throw std::runtime_error(
                "star_detect_fused_pixel_components: blurred image has zero range");
        }
        const double blur_mean = blur_sum / static_cast<double>(total);
        normalize_kernel<<<blocks, threads, 0, stream>>>(image.get(), image.get(), blur_mean,
                                                         blur_range, total);
        throw_if_cuda_failed(cudaGetLastError(),
                             "star_detect_fused_pixel_components normalize launch");

        // Linear resizing between equal shapes samples every pixel centre with
        // zero weight, so it is an exact copy and is skipped in both directions.
        if (small_height == height && small_width == width) {
            small_blur = std::move(image);
        } else {
            small_blur.allocate(
                small_size, "star_detect_fused_pixel_components cudaMalloc small blur", &workspace);
            resize_linear_kernel<<<small_blocks, threads, 0, stream>>>(
                image.get(), small_blur.get(), height, width, small_height, small_width);
            throw_if_cuda_failed(cudaGetLastError(),
                                 "star_detect_fused_pixel_components blur resize launch");
            image.reset();
        }

        DeviceImage rec_small = wavelet_dec_rec_device(
            std::move(small_blur), small_height, small_width, level, threads, &workspace, stream);

        if (rec_small.h == height && rec_small.w == width) {
            img_rec = std::move(rec_small.data);
            apply_mask_in_place_kernel<<<blocks, threads, 0, stream>>>(img_rec.get(), mask.get(),
                                                                       total);
            throw_if_cuda_failed(cudaGetLastError(),
                                 "star_detect_fused_pixel_components mask apply launch");
        } else {
            img_rec.allocate(plane_size, "star_detect_fused_pixel_components cudaMalloc img_rec",
                             &workspace);
            resize_linear_mask_kernel<<<blocks, threads, 0, stream>>>(
                rec_small.data.get(), mask.get(), img_rec.get(), rec_small.h, rec_small.w, height,
                width);
            throw_if_cuda_failed(cudaGetLastError(),
                                 "star_detect_fused_pixel_components resize up launch");
            rec_small.data.reset();
        }

        select_histogram.allocate(kSelectBins,
                                  "star_detect_fused_pixel_components cudaMalloc select histogram",
                                  &workspace);
        SelectWorkspace select;
        select.values = img_rec.get();
        select.mask = mask.get();
        select.total = total;
        select.histogram_device = select_histogram.get();
        select.histogram_host = histogram_host;
        select.count_device = count.get();
        select.workspace = &workspace;
        select.stream = stream;
        select.threads = threads;
        select.grid = std::max(1, std::min(blocks, 4096));
        const double threshold = masked_percentile_995(select);
        select_histogram.reset();

        eroded.allocate(plane_size, "star_detect_fused_pixel_components cudaMalloc eroded",
                        &workspace);
        bw.allocate(plane_size, "star_detect_fused_pixel_components cudaMalloc bw", &workspace);
        erode_threshold_kernel<<<blocks, threads, 0, stream>>>(
            img_rec.get(), mask.get(), eroded.get(), height, width, threshold);
        throw_if_cuda_failed(cudaGetLastError(), "star_detect_fused_pixel_components erode launch");
        dilate_kernel<<<blocks, threads, 0, stream>>>(eroded.get(), bw.get(), height, width);
        throw_if_cuda_failed(cudaGetLastError(),
                             "star_detect_fused_pixel_components dilate launch");
        eroded.reset();
        mask.reset();
        hnw::cuda::staged_download(binary_mask_host, bw.get(), mask_bytes, staging, &staging_slots,
                                   stream);

        throw_if_cuda_failed(cudaMemsetAsync(count.get(), 0, sizeof(int), stream),
                             "star_detect_fused_pixel_components cudaMemset foreground total");
        count_foreground_kernel<<<blocks, threads, 0, stream>>>(bw.get(), count.get(), total);
        throw_if_cuda_failed(cudaGetLastError(),
                             "star_detect_fused_pixel_components foreground count launch");
        throw_if_cuda_failed(
            cudaMemcpyAsync(scalar_int, count.get(), sizeof(int), cudaMemcpyDeviceToHost, stream),
            "star_detect_fused_pixel_components cudaMemcpy foreground total");
        throw_if_cuda_failed(cudaStreamSynchronize(stream),
                             "star_detect_fused_pixel_components foreground total sync");
        const int foreground_count = *scalar_int;
        if (foreground_count <= 0) {
            return true;
        }
        if (foreground_count > total / hnw::star_detect::kMaxForegroundDivisor) {
            throw hnw::StarDetectCapacityError("star_detect_fused_pixel_components: "
                                               "foreground too dense for GPU CC");
        }

        foreground_indices.allocate(
            static_cast<size_t>(foreground_count),
            "star_detect_fused_pixel_components cudaMalloc foreground indices", &workspace);
        throw_if_cuda_failed(cudaMemsetAsync(count.get(), 0, sizeof(int), stream),
                             "star_detect_fused_pixel_components cudaMemset foreground count");
        compact_foreground_indices_kernel<<<blocks, threads, 0, stream>>>(
            bw.get(), foreground_indices.get(), count.get(), total);
        throw_if_cuda_failed(cudaGetLastError(),
                             "star_detect_fused_pixel_components foreground compact launch");

        labels_a.allocate(plane_size, "star_detect_fused_pixel_components cudaMalloc labels a",
                          &workspace);
        labels_b.allocate(plane_size, "star_detect_fused_pixel_components cudaMalloc labels b",
                          &workspace);
        changed.allocate(1, "star_detect_fused_pixel_components cudaMalloc changed", &workspace);
        init_component_labels_kernel<<<blocks, threads, 0, stream>>>(bw.get(), labels_a.get(),
                                                                     total);
        throw_if_cuda_failed(cudaGetLastError(),
                             "star_detect_fused_pixel_components label init launch");
        throw_if_cuda_failed(cudaMemsetAsync(labels_b.get(), 0, plane_size * sizeof(int), stream),
                             "star_detect_fused_pixel_components cudaMemset labels b");
        bw.reset();

        bool converged = false;
        const int max_label_iterations =
            std::min(std::max(height, width), hnw::star_detect::kMaxLabelIterations);
        const int component_blocks = (foreground_count + threads - 1) / threads;
        for (int iter = 0; iter < max_label_iterations; ++iter) {
            throw_if_cuda_failed(cudaMemsetAsync(changed.get(), 0, sizeof(int), stream),
                                 "star_detect_fused_pixel_components cudaMemset changed");
            propagate_foreground_component_labels_kernel<<<component_blocks, threads, 0, stream>>>(
                foreground_indices.get(), foreground_count, labels_a.get(), labels_b.get(),
                changed.get(), height, width);
            throw_if_cuda_failed(cudaGetLastError(),
                                 "star_detect_fused_pixel_components label propagation launch");
            throw_if_cuda_failed(cudaMemcpyAsync(scalar_int, changed.get(), sizeof(int),
                                                 cudaMemcpyDeviceToHost, stream),
                                 "star_detect_fused_pixel_components cudaMemcpy changed");
            throw_if_cuda_failed(cudaStreamSynchronize(stream),
                                 "star_detect_fused_pixel_components label propagation sync");
            const int changed_count = *scalar_int;
            std::swap(labels_a, labels_b);
            if (changed_count == 0) {
                converged = true;
                break;
            }
        }
        if (!converged) {
            throw hnw::StarDetectCapacityError(
                "star_detect_fused_pixel_components: GPU CC did not converge");
        }

        if (foreground_count > std::numeric_limits<int>::max() / 2) {
            throw hnw::StarDetectCapacityError(
                "star_detect_fused_pixel_components: too many foreground pixels");
        }
        const int hash_capacity = next_power_of_two_int(
            std::max(hnw::star_detect::kMinHashCapacity,
                     foreground_count * hnw::star_detect::kHashCapacityMultiplier));
        const int hash_blocks = (hash_capacity + threads - 1) / threads;
        keys.allocate(hash_capacity, "star_detect_fused_pixel_components cudaMalloc hash keys",
                      &workspace);
        counts.allocate(hash_capacity, "star_detect_fused_pixel_components cudaMalloc hash counts",
                        &workspace);
        overflow.allocate(1, "star_detect_fused_pixel_components cudaMalloc overflow", &workspace);
        sum_x.allocate(hash_capacity, "star_detect_fused_pixel_components cudaMalloc sum_x",
                       &workspace);
        sum_y.allocate(hash_capacity, "star_detect_fused_pixel_components cudaMalloc sum_y",
                       &workspace);
        sum_intensity.allocate(hash_capacity,
                               "star_detect_fused_pixel_components cudaMalloc sum_intensity",
                               &workspace);

        throw_if_cuda_failed(cudaMemsetAsync(keys.get(), 0, hash_capacity * sizeof(int), stream),
                             "star_detect_fused_pixel_components cudaMemset keys");
        throw_if_cuda_failed(cudaMemsetAsync(counts.get(), 0, hash_capacity * sizeof(int), stream),
                             "star_detect_fused_pixel_components cudaMemset counts");
        throw_if_cuda_failed(cudaMemsetAsync(overflow.get(), 0, sizeof(int), stream),
                             "star_detect_fused_pixel_components cudaMemset overflow");
        throw_if_cuda_failed(
            cudaMemsetAsync(sum_x.get(), 0, hash_capacity * sizeof(double), stream),
            "star_detect_fused_pixel_components cudaMemset sum_x");
        throw_if_cuda_failed(
            cudaMemsetAsync(sum_y.get(), 0, hash_capacity * sizeof(double), stream),
            "star_detect_fused_pixel_components cudaMemset sum_y");
        throw_if_cuda_failed(
            cudaMemsetAsync(sum_intensity.get(), 0, hash_capacity * sizeof(double), stream),
            "star_detect_fused_pixel_components cudaMemset sum_intensity");

        const int foreground_blocks = (foreground_count + threads - 1) / threads;
        accumulate_component_stats_kernel<<<foreground_blocks, threads, 0, stream>>>(
            foreground_indices.get(), foreground_count, labels_a.get(), img_rec.get(), keys.get(),
            counts.get(), sum_x.get(), sum_y.get(), sum_intensity.get(), overflow.get(), width,
            hash_capacity);
        throw_if_cuda_failed(cudaGetLastError(), "star_detect_fused_pixel_components stats launch");
        foreground_indices.reset();
        labels_a.reset();
        labels_b.reset();
        img_rec.reset();

        throw_if_cuda_failed(cudaMemcpyAsync(scalar_int, overflow.get(), sizeof(int),
                                             cudaMemcpyDeviceToHost, stream),
                             "star_detect_fused_pixel_components cudaMemcpy overflow");
        throw_if_cuda_failed(cudaStreamSynchronize(stream),
                             "star_detect_fused_pixel_components overflow sync");
        const int overflow_flag = *scalar_int;
        if (overflow_flag != 0) {
            throw hnw::StarDetectCapacityError(
                "star_detect_fused_pixel_components: component hash table overflow");
        }

        throw_if_cuda_failed(cudaMemsetAsync(count.get(), 0, sizeof(int), stream),
                             "star_detect_fused_pixel_components cudaMemset output count");
        count_component_outputs_kernel<<<hash_blocks, threads, 0, stream>>>(
            keys.get(), counts.get(), count.get(), hash_capacity);
        throw_if_cuda_failed(cudaGetLastError(),
                             "star_detect_fused_pixel_components output count launch");

        throw_if_cuda_failed(
            cudaMemcpyAsync(scalar_int, count.get(), sizeof(int), cudaMemcpyDeviceToHost, stream),
            "star_detect_fused_pixel_components cudaMemcpy output count");
        throw_if_cuda_failed(cudaStreamSynchronize(stream),
                             "star_detect_fused_pixel_components output count sync");
        const int output_count = *scalar_int;
        if (output_count <= 0) {
            return true;
        }

        out_positions.allocate(static_cast<size_t>(output_count) * 2,
                               "star_detect_fused_pixel_components cudaMalloc output positions",
                               &workspace);
        out_intensities.allocate(static_cast<size_t>(output_count),
                                 "star_detect_fused_pixel_components cudaMalloc output intensities",
                                 &workspace);

        throw_if_cuda_failed(cudaMemsetAsync(count.get(), 0, sizeof(int), stream),
                             "star_detect_fused_pixel_components cudaMemset output index");
        fill_component_outputs_kernel<<<hash_blocks, threads, 0, stream>>>(
            keys.get(), counts.get(), sum_x.get(), sum_y.get(), sum_intensity.get(),
            out_positions.get(), out_intensities.get(), count.get(), hash_capacity);
        throw_if_cuda_failed(cudaGetLastError(),
                             "star_detect_fused_pixel_components output fill launch");

        positions_xy_host->resize(static_cast<size_t>(output_count) * 2);
        intensities_host->resize(static_cast<size_t>(output_count));
        const size_t positions_bytes = positions_xy_host->size() * sizeof(double);
        const size_t intensities_bytes = intensities_host->size() * sizeof(double);
        throw_if_cuda_failed(cudaMemcpyAsync(positions_xy_host->data(), out_positions.get(),
                                             positions_bytes, cudaMemcpyDeviceToHost, stream),
                             "star_detect_fused_pixel_components cudaMemcpy positions");
        throw_if_cuda_failed(cudaMemcpyAsync(intensities_host->data(), out_intensities.get(),
                                             intensities_bytes, cudaMemcpyDeviceToHost, stream),
                             "star_detect_fused_pixel_components cudaMemcpy intensities");
        throw_if_cuda_failed(cudaStreamSynchronize(stream),
                             "star_detect_fused_pixel_components final sync");
        return true;
    } catch (...) {
        workspace.reset_after_error();
        throw;
    }
}
