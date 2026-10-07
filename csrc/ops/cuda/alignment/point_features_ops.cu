#include "point_features_ops.h"

#include "common/compat.h"
#include "common/cuda_host_io_workspace.cuh"
#include "common/cuda_host_staging.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {

constexpr int kThreads = 128;
constexpr int kHistogramWarps = 8;
constexpr int kBins = 120;
constexpr int kBinsPerLane = (kBins + 31) / 32;
constexpr double kPi = 3.14159265358979323846;
constexpr double kBinStep = 3.14159265358979323846 / 60.0;

struct GridView {
    const double* unit = nullptr;
    const long long* cell_start = nullptr;
    const long long* cell_points = nullptr;
    double min[3] = {0.0, 0.0, 0.0};
    double cell = 0.0;
    long long dims[3] = {1, 1, 1};
    bool usable = false;
};

// Explicitly rounded arithmetic mirrors the CPU backend's expression order
// (no FMA contraction): neighbour ranking is exact, and descriptors differ
// only through the device acos/atan2/exp rounding.
__device__ inline double dot3_rn(const double* a, const double* b) {
    return __dadd_rn(__dadd_rn(__dmul_rn(a[0], b[0]), __dmul_rn(a[1], b[1])),
                     __dmul_rn(a[2], b[2]));
}

__device__ inline double clamp_unit(const double value) {
    const double upper = value < 1.0 ? value : 1.0;
    return -1.0 < upper ? upper : -1.0;
}

// Matches np.inner(v, make_cross_matrix(base)).
__device__ inline void inner_with_cross_matrix(const double* v, const double* base, double* out) {
    out[0] = __dadd_rn(__dmul_rn(-v[1], base[2]), __dmul_rn(v[2], base[1]));
    out[1] = __dsub_rn(__dmul_rn(v[0], base[2]), __dmul_rn(v[2], base[0]));
    out[2] = __dadd_rn(__dmul_rn(-v[0], base[1]), __dmul_rn(v[1], base[0]));
}

__device__ inline void normalize3(double* vec) {
    const double norm = __dsqrt_rn(dot3_rn(vec, vec));
    if (norm == 0.0 || !isfinite(norm)) {
        vec[0] = 0.0;
        vec[1] = 0.0;
        vec[2] = 0.0;
        return;
    }
    for (int axis = 0; axis < 3; ++axis) {
        vec[axis] = __ddiv_rn(vec[axis], norm);
    }
}

// Candidates farther (in squared unit distance) than every pooled neighbour by
// this margin have a cosine lower by >= 5e-13, far above rounding, so they
// cannot enter the pool and skip the exact cosine.
constexpr double kDistance2Margin = 1e-12;

// Relative gap below which two vol*rho keys count as tied: far above the few
// ulps by which device and host acos may differ.
constexpr double kKeyTieMargin = 1e-12;

// Pool ordered by similarity descending, then index ascending (a total order,
// so the kept prefix equals the CPU's partial sort).
struct NeighbourPool {
    double similarity[kPointFeatureMaxPool];
    double distance2[kPointFeatureMaxPool];
    int index[kPointFeatureMaxPool];
    int count = 0;
    int capacity = 0;
    double farthest2 = 0.0; // max distance2 once the pool is full

    __device__ bool full() const { return count == capacity; }

    __device__ static bool precedes(const double sa, const int ia, const double sb, const int ib) {
        return sa != sb ? sa > sb : ia < ib;
    }

    __device__ void offer(const double sim, const int j, const double d2) {
        if (full() && !precedes(sim, j, similarity[count - 1], index[count - 1])) {
            return;
        }
        int pos = count < capacity ? count++ : capacity - 1;
        while (pos > 0 && precedes(sim, j, similarity[pos - 1], index[pos - 1])) {
            similarity[pos] = similarity[pos - 1];
            distance2[pos] = distance2[pos - 1];
            index[pos] = index[pos - 1];
            --pos;
        }
        similarity[pos] = sim;
        distance2[pos] = d2;
        index[pos] = j;
        if (full()) {
            farthest2 = 0.0;
            for (int m = 0; m < count; ++m) {
                farthest2 = fmax(farthest2, distance2[m]);
            }
        }
    }
};

__global__ void norms_kernel(const double* vec, const int n, double* norms) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const double* v = vec + static_cast<size_t>(i) * 3;
        norms[i] = __dsqrt_rn(dot3_rn(v, v));
    }
}

// Per point: exact top-`pool` neighbours by cosine similarity, the stable
// vol*rho ordering of that pool, then per selected neighbour the angle theta,
// the Gaussian denominator 2*sigma^2 and the weight vol*rho^2/sigma.
__global__ void select_neighbours_kernel(const double* vec, const double* vol, const double* norms,
                                         const int n, const int k, const int pool,
                                         const GridView grid, double* params, int* ambiguous) {
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread >= n) {
        return;
    }
    // Grid order keeps a warp's queries, and so its candidate cells, close.
    const int i = grid.usable ? static_cast<int>(grid.cell_points[thread]) : thread;
    const double* v0 = vec + static_cast<size_t>(i) * 3;
    const double n0 = norms[i];
    NeighbourPool best;
    best.capacity = pool;
    auto offer = [&](const long long j, const double* u0) {
        double d2 = 0.0;
        if (u0 != nullptr) {
            const double* u1 = grid.unit + j * 3;
            const double dx = __dsub_rn(u0[0], u1[0]);
            const double dy = __dsub_rn(u0[1], u1[1]);
            const double dz = __dsub_rn(u0[2], u1[2]);
            d2 = __dadd_rn(__dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy)), __dmul_rn(dz, dz));
            if (best.full() && d2 > best.farthest2 + kDistance2Margin) {
                return;
            }
        }
        const double nj = norms[j];
        const double cosine =
            n0 == 0.0 || nj == 0.0 ? 0.0 : __ddiv_rn(dot3_rn(v0, vec + j * 3), __dmul_rn(n0, nj));
        best.offer(clamp_unit(cosine), static_cast<int>(j), d2);
    };

    if (!grid.usable) {
        for (long long j = 0; j < n; ++j) {
            offer(j, nullptr);
        }
    } else {
        // Shells stop once every pooled neighbour is closer than any unvisited
        // cell by the CPU grid's margin, so the pool equals a full scan's.
        const double* u0 = grid.unit + static_cast<size_t>(i) * 3;
        long long q[3];
        for (int axis = 0; axis < 3; ++axis) {
            const long long c =
                static_cast<long long>(__ddiv_rn(__dsub_rn(u0[axis], grid.min[axis]), grid.cell));
            q[axis] = c < 0 ? 0 : (c > grid.dims[axis] - 1 ? grid.dims[axis] - 1 : c);
        }
        for (long long r = 0;; ++r) {
            long long lo[3];
            long long hi[3];
            bool covers_grid = true;
            for (int axis = 0; axis < 3; ++axis) {
                lo[axis] = q[axis] - r < 0 ? 0 : q[axis] - r;
                hi[axis] = q[axis] + r > grid.dims[axis] - 1 ? grid.dims[axis] - 1 : q[axis] + r;
                covers_grid = covers_grid && q[axis] - r <= 0 && q[axis] + r >= grid.dims[axis] - 1;
            }
            for (long long z = lo[2]; z <= hi[2]; ++z) {
                for (long long y = lo[1]; y <= hi[1]; ++y) {
                    for (long long x = lo[0]; x <= hi[0]; ++x) {
                        const long long ring =
                            max(max(llabs(x - q[0]), llabs(y - q[1])), llabs(z - q[2]));
                        if (ring != r) {
                            continue;
                        }
                        const long long cell = (z * grid.dims[1] + y) * grid.dims[0] + x;
                        for (long long slot = grid.cell_start[cell];
                             slot < grid.cell_start[cell + 1]; ++slot) {
                            offer(grid.cell_points[slot], u0);
                        }
                    }
                }
            }
            if (covers_grid) {
                break;
            }
            if (best.full() &&
                __dsqrt_rn(best.farthest2) + hnw::alignment::DirectionGrid::kGridMargin <
                    static_cast<double>(r) * grid.cell) {
                break;
            }
        }
    }

    // Stable vol*rho order; the keys reuse the distance slots.
    double rho[kPointFeatureMaxPool];
    int order[kPointFeatureMaxPool];
    for (int m = 0; m < pool; ++m) {
        rho[m] = acos(best.similarity[m]);
        best.distance2[m] = __dmul_rn(vol[best.index[m]], rho[m]);
        int pos = m;
        while (pos > 0 && best.distance2[m] > best.distance2[order[pos - 1]]) {
            order[pos] = order[pos - 1];
            --pos;
        }
        order[pos] = m;
    }
    // The first key picks the reference angle and the k-th boundary picks the
    // neighbours. A near tie there could order differently under the host
    // acos, so the host recomputes the call on the CPU.
    const auto near_tie = [&](const int a, const int b) {
        const double sa = best.similarity[a];
        const double sb = best.similarity[b];
        const double va = vol[best.index[a]];
        const double vb = vol[best.index[b]];
        if ((sa == sb && va == vb) || (sa == 1.0 && sb == 1.0)) {
            return false; // identical keys (acos(1) == 0) on every backend
        }
        const double ka = best.distance2[a];
        const double kb = best.distance2[b];
        return !(fabs(ka - kb) > kKeyTieMargin * fmax(fabs(ka), fabs(kb)));
    };
    if ((k > 1 && near_tie(order[0], order[1])) || (pool > k && near_tie(order[k - 1], order[k]))) {
        *ambiguous = 1;
    }

    double angle0[3] = {0.0, 0.0, 0.0};
    double* out = params + static_cast<size_t>(i) * k * 3;
    for (int jj = 0; jj < k; ++jj) {
        const int pool_pos = order[jj];
        const int src = best.index[pool_pos];
        double angle[3];
        inner_with_cross_matrix(vec + static_cast<size_t>(src) * 3, v0, angle);
        normalize3(angle);
        if (jj == 0) {
            angle0[0] = angle[0];
            angle0[1] = angle[1];
            angle0[2] = angle[2];
        }
        double cr[3];
        inner_with_cross_matrix(angle, angle0, cr);
        const double s_norm = __dsqrt_rn(dot3_rn(cr, cr));
        const double sign_dot = dot3_rn(cr, v0);
        const double s =
            __dmul_rn(s_norm, static_cast<double>((sign_dot > 0.0) - (sign_dot < 0.0)));
        const double theta = atan2(s, dot3_rn(angle, angle0));
        const double r = rho[pool_pos];
        const double sigma = __dadd_rn(__dmul_rn(2.5, exp(__dmul_rn(-r, 100.0))), 0.04);
        out[jj * 3] = theta;
        out[jj * 3 + 1] = __dmul_rn(__dmul_rn(2.0, sigma), sigma);
        out[jj * 3 + 2] = __ddiv_rn(__dmul_rn(__dmul_rn(vol[src], r), r), sigma);
    }
}

// One warp per point; lanes own bins and accumulate neighbours in order, then
// lane 0 forms the norm over bins in order, as the CPU loop does.
__global__ void histogram_kernel(const double* params, const int n, const int k, double* out) {
    __shared__ double rows[kHistogramWarps][kBins];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int i = blockIdx.x * kHistogramWarps + warp;
    if (i >= n) {
        return;
    }
    double acc[kBinsPerLane];
    for (int t = 0; t < kBinsPerLane; ++t) {
        acc[t] = 0.0;
    }
    const double* point = params + static_cast<size_t>(i) * k * 3;
    for (int jj = 0; jj < k; ++jj) {
        const double theta = point[jj * 3];
        const double denominator = point[jj * 3 + 1];
        const double scale = point[jj * 3 + 2];
        for (int t = 0; t < kBinsPerLane; ++t) {
            const int bin = lane + 32 * t;
            if (bin < kBins) {
                const double fx = __dadd_rn(-kPi, __dmul_rn(static_cast<double>(bin), kBinStep));
                const double delta = __dsub_rn(theta, fx);
                const double weight = exp(__ddiv_rn(-__dmul_rn(delta, delta), denominator));
                acc[t] = __dadd_rn(acc[t], __dmul_rn(weight, scale));
            }
        }
    }
    for (int t = 0; t < kBinsPerLane; ++t) {
        const int bin = lane + 32 * t;
        if (bin < kBins) {
            rows[warp][bin] = acc[t];
        }
    }
    __syncwarp();
    double norm = 0.0;
    if (lane == 0) {
        for (int bin = 0; bin < kBins; ++bin) {
            norm = __dadd_rn(norm, __dmul_rn(rows[warp][bin], rows[warp][bin]));
        }
        norm = __dsqrt_rn(norm);
    }
    norm = __shfl_sync(0xffffffffu, norm, 0);
    const bool normalize = norm > 0.0 && isfinite(norm);
    for (int t = 0; t < kBinsPerLane; ++t) {
        const int bin = lane + 32 * t;
        if (bin < kBins) {
            out[static_cast<size_t>(i) * kBins + bin] =
                normalize ? __ddiv_rn(acc[t], norm) : acc[t];
        }
    }
}

template <typename T>
T* upload(hnw::cuda::HostIoWorkspaceSession* workspace, const T* host, const size_t count,
          const cudaStream_t stream, const char* context) {
    auto* device = static_cast<T*>(workspace->device_buffer(count * sizeof(T), context));
    hnw::cuda::throw_if_failed(
        cudaMemcpyAsync(device, host, count * sizeof(T), cudaMemcpyHostToDevice, stream), context);
    return device;
}

} // namespace

bool launch_extract_point_features_cuda(const double* vec, const double* vol, const int64_t n,
                                        const int k, const int pool,
                                        const hnw::alignment::DirectionGrid& grid, double* out) {
    if (n > std::numeric_limits<int>::max() / kBins) {
        throw std::invalid_argument("extract_point_features: too many points for CUDA");
    }
    auto workspace = hnw::cuda::acquire_host_io_workspace("extract_point_features cudaGetDevice");
    try {
        const cudaStream_t stream = workspace.stream();
        const auto count = static_cast<size_t>(n);
        const double* vec_device =
            upload(&workspace, vec, count * 3, stream, "extract_point_features cudaMalloc vec");
        const double* vol_device =
            upload(&workspace, vol, count, stream, "extract_point_features cudaMalloc vol");
        auto* norms = static_cast<double*>(workspace.device_buffer(
            count * sizeof(double), "extract_point_features cudaMalloc norms"));

        GridView view;
        if (grid.usable()) {
            view.unit = upload(&workspace, grid.unit().data(), grid.unit().size(), stream,
                               "extract_point_features cudaMalloc unit");
            static_assert(sizeof(long long) == sizeof(int64_t), "int64 grid offsets");
            view.cell_start = upload(
                &workspace, reinterpret_cast<const long long*>(grid.cell_start().data()),
                grid.cell_start().size(), stream, "extract_point_features cudaMalloc cell_start");
            static_assert(sizeof(ssize_t) == sizeof(long long), "ssize_t grid points");
            view.cell_points = upload(
                &workspace, reinterpret_cast<const long long*>(grid.cell_points().data()),
                grid.cell_points().size(), stream, "extract_point_features cudaMalloc cell_points");
            for (int axis = 0; axis < 3; ++axis) {
                view.min[axis] = grid.min()[axis];
                view.dims[axis] = static_cast<long long>(grid.dims()[axis]);
            }
            view.cell = grid.cell();
            view.usable = true;
        }
        auto* params = static_cast<double*>(
            workspace.device_buffer(count * static_cast<size_t>(k) * 3 * sizeof(double),
                                    "extract_point_features cudaMalloc params"));
        auto* features = static_cast<double*>(workspace.device_buffer(
            count * kBins * sizeof(double), "extract_point_features cudaMalloc features"));
        auto* ambiguous_device = static_cast<int*>(
            workspace.device_buffer(sizeof(int), "extract_point_features cudaMalloc ambiguous"));
        hnw::cuda::throw_if_failed(cudaMemsetAsync(ambiguous_device, 0, sizeof(int), stream),
                                   "extract_point_features cudaMemset ambiguous");

        const int points = static_cast<int>(n);
        const int blocks = (points + kThreads - 1) / kThreads;
        norms_kernel<<<blocks, kThreads, 0, stream>>>(vec_device, points, norms);
        hnw::cuda::throw_if_failed(cudaGetLastError(), "extract_point_features norms launch");
        select_neighbours_kernel<<<blocks, kThreads, 0, stream>>>(
            vec_device, vol_device, norms, points, k, pool, view, params, ambiguous_device);
        hnw::cuda::throw_if_failed(cudaGetLastError(), "extract_point_features select launch");
        int ambiguous = 0;
        hnw::cuda::throw_if_failed(cudaMemcpyAsync(&ambiguous, ambiguous_device, sizeof(int),
                                                   cudaMemcpyDeviceToHost, stream),
                                   "extract_point_features cudaMemcpy ambiguous");
        hnw::cuda::throw_if_failed(cudaStreamSynchronize(stream),
                                   "extract_point_features cudaStreamSynchronize");
        if (ambiguous != 0) {
            return false;
        }
        histogram_kernel<<<(points + kHistogramWarps - 1) / kHistogramWarps, kHistogramWarps * 32,
                           0, stream>>>(params, points, k, features);
        hnw::cuda::throw_if_failed(cudaGetLastError(), "extract_point_features histogram launch");
        // Pinned double-buffered staging: the pageable path is bound by the
        // driver's single-threaded staging copy.
        auto* staging = static_cast<unsigned char*>(workspace.pinned_buffer(
            2 * hnw::cuda::kStagingSlotBytes, "extract_point_features cudaMallocHost staging"));
        hnw::cuda::StagingSlots slots;
        hnw::cuda::staged_download(out, features, count * kBins * sizeof(double), staging, &slots,
                                   stream);
        hnw::cuda::throw_if_failed(cudaStreamSynchronize(stream),
                                   "extract_point_features cudaStreamSynchronize");
        return true;
    } catch (...) {
        workspace.reset_after_error();
        throw;
    }
}
