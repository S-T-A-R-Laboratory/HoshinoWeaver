#include "asterism_mutual_nearest_ops.h"

#include "common/compat.h"
#include "common/cuda_host_io_workspace.cuh"

#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <math_constants.h>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kFrameBlocks = 64;
constexpr int kFrameValues = 7; // lo[3], hi[3], non-finite flag
constexpr int kScanThreads = 1024;
constexpr int kScanItems = 4;
constexpr int64_t kScanTile = int64_t{kScanThreads} * kScanItems;
static_assert(hnw::asterism::kMaxGridCells <= kScanTile * kScanThreads,
              "tile sums must fit one scan block");

using TileScan = cub::BlockScan<int32_t, kScanThreads>;

struct DeviceFrame {
    double origin[3];
    double cell;
    long long dims[3];
};

struct TokenSet {
    const double* values;
    int count;
};

struct DeviceGrid {
    int32_t* cell_start = nullptr;
    int32_t* points = nullptr;
    double* coords = nullptr;
    int count = 0;
};

// Explicitly rounded operations reproduce the CPU backend's IEEE arithmetic
// (no FMA contraction), so both backends grid and pick identical tokens.
__device__ inline long long cell_coordinate(const double value, const double origin,
                                            const double cell) {
    return static_cast<long long>(floor(__ddiv_rn(__dsub_rn(value, origin), cell)));
}

__device__ inline int32_t token_cell(const double* point, const DeviceFrame& frame) {
    long long coords[3];
    for (int axis = 0; axis < 3; ++axis) {
        const long long q = cell_coordinate(point[axis], frame.origin[axis], frame.cell);
        coords[axis] = q < 0 ? 0 : (q > frame.dims[axis] - 1 ? frame.dims[axis] - 1 : q);
    }
    return static_cast<int32_t>((coords[2] * frame.dims[1] + coords[1]) * frame.dims[0] +
                                coords[0]);
}

__device__ inline double squared_distance_rn(const double* a, const double* b) {
    const double d0 = __dsub_rn(a[0], b[0]);
    const double d1 = __dsub_rn(a[1], b[1]);
    const double d2 = __dsub_rn(a[2], b[2]);
    double sum = __dadd_rn(0.0, __dmul_rn(d0, d0));
    sum = __dadd_rn(sum, __dmul_rn(d1, d1));
    return __dadd_rn(sum, __dmul_rn(d2, d2));
}

// Per-block extent of one token set (blockIdx.y); the host merges the
// partials. Min/max are order independent, matching the CPU extent exactly.
__global__ void frame_partials_kernel(const TokenSet set1, const TokenSet set2, double* partials) {
    const TokenSet set = blockIdx.y == 0 ? set1 : set2;
    double value[kFrameValues] = {CUDART_INF,  CUDART_INF,  CUDART_INF, -CUDART_INF,
                                  -CUDART_INF, -CUDART_INF, 0.0};
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < set.count;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        for (int axis = 0; axis < 3; ++axis) {
            const double v = set.values[i * 3 + axis];
            if (isfinite(v)) {
                value[axis] = fmin(value[axis], v);
                value[3 + axis] = fmax(value[3 + axis], v);
            } else {
                value[6] = 1.0;
            }
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        for (int k = 0; k < kFrameValues; ++k) {
            const double other = __shfl_down_sync(0xffffffffu, value[k], offset);
            value[k] = k < 3 ? fmin(value[k], other) : fmax(value[k], other);
        }
    }
    __shared__ double warp_values[kThreads / 32][kFrameValues];
    const int warp = threadIdx.x / 32;
    if (threadIdx.x % 32 == 0) {
        for (int k = 0; k < kFrameValues; ++k) {
            warp_values[warp][k] = value[k];
        }
    }
    __syncthreads();
    if (threadIdx.x < kFrameValues) {
        const int k = threadIdx.x;
        double merged = warp_values[0][k];
        for (int w = 1; w < kThreads / 32; ++w) {
            merged = k < 3 ? fmin(merged, warp_values[w][k]) : fmax(merged, warp_values[w][k]);
        }
        partials[(blockIdx.y * gridDim.x + blockIdx.x) * kFrameValues + k] = merged;
    }
}

__global__ void count_cells_kernel(const TokenSet set, const DeviceFrame frame,
                                   int32_t* cell_start) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < set.count) {
        atomicAdd(&cell_start[token_cell(set.values + static_cast<size_t>(i) * 3, frame) + 1], 1);
    }
}

// In-place exclusive scan of data[0, length): tiles scan locally and record
// their totals, one block scans the totals, then tiles add their offsets.
__global__ void __launch_bounds__(kScanThreads)
    scan_tiles_kernel(int32_t* data, const long long length, int32_t* tile_sums) {
    __shared__ typename TileScan::TempStorage temp;
    const long long base = blockIdx.x * kScanTile + threadIdx.x * kScanItems;
    int32_t items[kScanItems];
    int32_t sum = 0;
    for (int k = 0; k < kScanItems; ++k) {
        items[k] = base + k < length ? data[base + k] : 0;
        sum += items[k];
    }
    int32_t prefix = 0;
    int32_t total = 0;
    TileScan(temp).ExclusiveSum(sum, prefix, total);
    for (int k = 0; k < kScanItems; ++k) {
        if (base + k < length) {
            data[base + k] = prefix;
        }
        prefix += items[k];
    }
    if (threadIdx.x == 0) {
        tile_sums[blockIdx.x] = total;
    }
}

__global__ void __launch_bounds__(kScanThreads)
    scan_tile_sums_kernel(int32_t* tile_sums, const int tiles) {
    __shared__ typename TileScan::TempStorage temp;
    const int32_t value = static_cast<int>(threadIdx.x) < tiles ? tile_sums[threadIdx.x] : 0;
    int32_t prefix = 0;
    TileScan(temp).ExclusiveSum(value, prefix);
    if (static_cast<int>(threadIdx.x) < tiles) {
        tile_sums[threadIdx.x] = prefix;
    }
}

__global__ void add_tile_offsets_kernel(int32_t* data, const long long length,
                                        const int32_t* tile_offsets) {
    const int32_t offset = tile_offsets[blockIdx.x];
    const long long base = blockIdx.x * kScanTile;
    for (long long k = threadIdx.x; k < kScanTile && base + k < length; k += blockDim.x) {
        data[base + k] += offset;
    }
}

// cell_start[c + 1] holds the start of cell c; claiming slots advances it to
// the start of cell c + 1, leaving CSR offsets. Slot order within a cell is
// arbitrary; the nearest search breaks ties by index, so it does not matter.
__global__ void scatter_tokens_kernel(const TokenSet set, const DeviceFrame frame,
                                      int32_t* cell_start, int32_t* points, double* coords) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= set.count) {
        return;
    }
    const double* point = set.values + static_cast<size_t>(i) * 3;
    const int32_t slot = atomicAdd(&cell_start[token_cell(point, frame) + 1], 1);
    points[slot] = i;
    for (int axis = 0; axis < 3; ++axis) {
        coords[static_cast<size_t>(slot) * 3 + axis] = point[axis];
    }
}

// Queries run in slot order so neighbouring threads probe the same cells.
// Mirrors the CPU bounded_nearest, raising *ambiguous on a nearest-distance tie.
__global__ void bounded_nearest_kernel(const DeviceGrid queries, const DeviceGrid grid,
                                       const DeviceFrame frame, const double threshold,
                                       int32_t* nearest, int* ambiguous) {
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= queries.count) {
        return;
    }
    const double* query = queries.coords + static_cast<size_t>(slot) * 3;
    long long lo[3];
    long long hi[3];
    for (int axis = 0; axis < 3; ++axis) {
        const long long q = cell_coordinate(query[axis], frame.origin[axis], frame.cell);
        lo[axis] = q - 1 < 0 ? 0 : q - 1;
        hi[axis] = q + 1 > frame.dims[axis] - 1 ? frame.dims[axis] - 1 : q + 1;
    }
    int32_t best = -1;
    bool tied = false;
    double best_distance = CUDART_INF;
    for (long long z = lo[2]; z <= hi[2]; ++z) {
        for (long long y = lo[1]; y <= hi[1]; ++y) {
            const long long row = (z * frame.dims[1] + y) * frame.dims[0];
            const int32_t begin = grid.cell_start[row + lo[0]];
            const int32_t end = grid.cell_start[row + hi[0] + 1];
            for (int32_t s = begin; s < end; ++s) {
                const int32_t j = grid.points[s];
                const double distance =
                    squared_distance_rn(query, grid.coords + static_cast<size_t>(s) * 3);
                if (distance < best_distance) {
                    best_distance = distance;
                    best = j;
                    tied = false;
                } else if (distance == best_distance) {
                    tied = true;
                }
            }
        }
    }
    const bool within = best >= 0 && __dsqrt_rn(best_distance) <= threshold;
    if (within && tied) {
        *ambiguous = 1;
    }
    nearest[queries.points[slot]] = within ? best : -1;
}

__global__ void keep_mutual_kernel(int32_t* nearest12, const int n1, const int32_t* nearest21) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n1) {
        const int32_t j = nearest12[i];
        if (j >= 0 && nearest21[j] != i) {
            nearest12[i] = -1;
        }
    }
}

int blocks_for(const long long count, const int threads) {
    return static_cast<int>((count + threads - 1) / threads);
}

void check_launch(const char* context) {
    hnw::cuda::throw_if_failed(cudaGetLastError(), context);
}

template <typename T>
T* device_array(hnw::cuda::HostIoWorkspaceSession* workspace, const size_t count,
                const char* context) {
    return static_cast<T*>(workspace->device_buffer(count * sizeof(T), context));
}

DeviceGrid allocate_grid(hnw::cuda::HostIoWorkspaceSession* workspace, const int count,
                         const int64_t cells) {
    DeviceGrid grid;
    grid.cell_start = device_array<int32_t>(workspace, static_cast<size_t>(cells) + 1,
                                            "asterism_mutual_nearest cudaMalloc cell_start");
    grid.points = device_array<int32_t>(workspace, static_cast<size_t>(count),
                                        "asterism_mutual_nearest cudaMalloc points");
    grid.coords = device_array<double>(workspace, static_cast<size_t>(count) * 3,
                                       "asterism_mutual_nearest cudaMalloc coords");
    grid.count = count;
    return grid;
}

void build_grid(const TokenSet set, const DeviceFrame& frame, const int64_t cells,
                const DeviceGrid& grid, int32_t* tile_sums, const cudaStream_t stream) {
    const int tiles = static_cast<int>((cells + kScanTile - 1) / kScanTile);
    hnw::cuda::throw_if_failed(cudaMemsetAsync(grid.cell_start, 0,
                                               (static_cast<size_t>(cells) + 1) * sizeof(int32_t),
                                               stream),
                               "asterism_mutual_nearest cudaMemset cell_start");
    count_cells_kernel<<<blocks_for(set.count, kThreads), kThreads, 0, stream>>>(set, frame,
                                                                                 grid.cell_start);
    check_launch("asterism_mutual_nearest count launch");
    scan_tiles_kernel<<<tiles, kScanThreads, 0, stream>>>(grid.cell_start + 1, cells, tile_sums);
    check_launch("asterism_mutual_nearest scan launch");
    scan_tile_sums_kernel<<<1, kScanThreads, 0, stream>>>(tile_sums, tiles);
    check_launch("asterism_mutual_nearest scan launch");
    add_tile_offsets_kernel<<<tiles, kThreads, 0, stream>>>(grid.cell_start + 1, cells, tile_sums);
    check_launch("asterism_mutual_nearest scan launch");
    scatter_tokens_kernel<<<blocks_for(set.count, kThreads), kThreads, 0, stream>>>(
        set, frame, grid.cell_start, grid.points, grid.coords);
    check_launch("asterism_mutual_nearest scatter launch");
}

void run_direction(const DeviceGrid& queries, const DeviceGrid& grid, const DeviceFrame& frame,
                   const double threshold, int32_t* nearest, int* ambiguous,
                   const cudaStream_t stream) {
    bounded_nearest_kernel<<<blocks_for(queries.count, kThreads), kThreads, 0, stream>>>(
        queries, grid, frame, threshold, nearest, ambiguous);
    check_launch("asterism_mutual_nearest kernel launch");
}

// Merges the per-block extents; false when any value is non-finite.
bool merge_extent(const std::vector<double>& partials, double lo[3], double hi[3]) {
    for (int axis = 0; axis < 3; ++axis) {
        lo[axis] = std::numeric_limits<double>::infinity();
        hi[axis] = -std::numeric_limits<double>::infinity();
    }
    for (size_t block = 0; block < partials.size() / kFrameValues; ++block) {
        const double* value = partials.data() + block * kFrameValues;
        if (value[6] != 0.0) {
            return false;
        }
        for (int axis = 0; axis < 3; ++axis) {
            lo[axis] = std::min(lo[axis], value[axis]);
            hi[axis] = std::max(hi[axis], value[3 + axis]);
        }
    }
    return true;
}

} // namespace

bool launch_asterism_mutual_nearest_cuda(const double* values1, const int64_t n1,
                                         const double* values2, const int64_t n2,
                                         const double threshold, std::vector<int32_t>* mutual12) {
    auto workspace = hnw::cuda::acquire_host_io_workspace("asterism_mutual_nearest cudaGetDevice");
    try {
        const cudaStream_t stream = workspace.stream();
        const TokenSet set1{device_array<double>(&workspace, static_cast<size_t>(n1) * 3,
                                                 "asterism_mutual_nearest cudaMalloc values1"),
                            static_cast<int>(n1)};
        const TokenSet set2{device_array<double>(&workspace, static_cast<size_t>(n2) * 3,
                                                 "asterism_mutual_nearest cudaMalloc values2"),
                            static_cast<int>(n2)};
        hnw::cuda::throw_if_failed(cudaMemcpyAsync(const_cast<double*>(set1.values), values1,
                                                   static_cast<size_t>(n1) * 3 * sizeof(double),
                                                   cudaMemcpyHostToDevice, stream),
                                   "asterism_mutual_nearest cudaMemcpy values1");
        hnw::cuda::throw_if_failed(cudaMemcpyAsync(const_cast<double*>(set2.values), values2,
                                                   static_cast<size_t>(n2) * 3 * sizeof(double),
                                                   cudaMemcpyHostToDevice, stream),
                                   "asterism_mutual_nearest cudaMemcpy values2");

        std::vector<double> partials(static_cast<size_t>(2) * kFrameBlocks * kFrameValues);
        auto* partials_device = device_array<double>(&workspace, partials.size(),
                                                     "asterism_mutual_nearest cudaMalloc extent");
        frame_partials_kernel<<<dim3(kFrameBlocks, 2), kThreads, 0, stream>>>(set1, set2,
                                                                              partials_device);
        check_launch("asterism_mutual_nearest extent launch");
        hnw::cuda::throw_if_failed(cudaMemcpyAsync(partials.data(), partials_device,
                                                   partials.size() * sizeof(double),
                                                   cudaMemcpyDeviceToHost, stream),
                                   "asterism_mutual_nearest cudaMemcpy extent");
        hnw::cuda::throw_if_failed(cudaStreamSynchronize(stream),
                                   "asterism_mutual_nearest cudaStreamSynchronize");

        double lo[3];
        double hi[3];
        hnw::asterism::GridFrame frame;
        if (!merge_extent(partials, lo, hi) ||
            !hnw::asterism::finish_token_grid_frame(lo, hi, threshold, &frame)) {
            return false;
        }
        const int64_t cells = frame.dims[0] * frame.dims[1] * frame.dims[2];
        DeviceFrame device_frame;
        for (int axis = 0; axis < 3; ++axis) {
            device_frame.origin[axis] = frame.origin[axis];
            device_frame.dims[axis] = static_cast<long long>(frame.dims[axis]);
        }
        device_frame.cell = frame.cell;

        const DeviceGrid grid1 = allocate_grid(&workspace, set1.count, cells);
        const DeviceGrid grid2 = allocate_grid(&workspace, set2.count, cells);
        auto* tile_sums = device_array<int32_t>(
            &workspace, static_cast<size_t>((cells + kScanTile - 1) / kScanTile),
            "asterism_mutual_nearest cudaMalloc tile sums");
        auto* nearest12 = device_array<int32_t>(&workspace, static_cast<size_t>(n1),
                                                "asterism_mutual_nearest cudaMalloc nearest12");
        auto* nearest21 = device_array<int32_t>(&workspace, static_cast<size_t>(n2),
                                                "asterism_mutual_nearest cudaMalloc nearest21");
        auto* ambiguous_device =
            device_array<int>(&workspace, 1, "asterism_mutual_nearest cudaMalloc ambiguous");
        hnw::cuda::throw_if_failed(cudaMemsetAsync(ambiguous_device, 0, sizeof(int), stream),
                                   "asterism_mutual_nearest cudaMemset ambiguous");
        build_grid(set1, device_frame, cells, grid1, tile_sums, stream);
        build_grid(set2, device_frame, cells, grid2, tile_sums, stream);
        run_direction(grid1, grid2, device_frame, threshold, nearest12, ambiguous_device, stream);
        run_direction(grid2, grid1, device_frame, threshold, nearest21, ambiguous_device, stream);
        keep_mutual_kernel<<<blocks_for(n1, kThreads), kThreads, 0, stream>>>(
            nearest12, static_cast<int>(n1), nearest21);
        check_launch("asterism_mutual_nearest mutual launch");

        mutual12->resize(static_cast<size_t>(n1));
        hnw::cuda::throw_if_failed(cudaMemcpyAsync(mutual12->data(), nearest12,
                                                   mutual12->size() * sizeof(int32_t),
                                                   cudaMemcpyDeviceToHost, stream),
                                   "asterism_mutual_nearest cudaMemcpy mutual");
        int ambiguous = 0;
        hnw::cuda::throw_if_failed(cudaMemcpyAsync(&ambiguous, ambiguous_device, sizeof(int),
                                                   cudaMemcpyDeviceToHost, stream),
                                   "asterism_mutual_nearest cudaMemcpy ambiguous");
        hnw::cuda::throw_if_failed(cudaStreamSynchronize(stream),
                                   "asterism_mutual_nearest cudaStreamSynchronize");
        return ambiguous == 0;
    } catch (...) {
        workspace.reset_after_error();
        throw;
    }
}
