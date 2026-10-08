#include "asterism_mutual_nearest_ops.h"

#include "common/cpu_compat.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace hnw::asterism {

namespace {

// Edge slightly above the threshold keeps every within-threshold neighbour in
// the adjacent cells despite rounding in the cell-coordinate division.
constexpr double kCellMargin = 1e-9;
constexpr int64_t kParallelTokens = 4096;
constexpr int kLightPassThreads = 8;

// Memory-bound per-token passes saturate with a few threads; a full team
// only adds fork/join cost and straggler stalls on busy hosts.
int light_pass_threads(const int64_t n) {
#if defined(_OPENMP)
    return n >= kParallelTokens ? std::min(omp_get_max_threads(), kLightPassThreads) : 1;
#else
    (void)n;
    return 1;
#endif
}

struct Extent {
    double lo[3] = {std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::infinity()};
    double hi[3] = {-std::numeric_limits<double>::infinity(),
                    -std::numeric_limits<double>::infinity(),
                    -std::numeric_limits<double>::infinity()};
    bool finite = true;
};

// Min/max are order independent, so the per-thread partials merge to the
// serial result.
void accumulate_extent(const double* values, const int64_t n, Extent* extent) {
    const int threads = light_pass_threads(n);
    (void)threads;
#if defined(_OPENMP)
#pragma omp parallel num_threads(threads)
#endif
    {
        // Thread-local partials start empty; `extent` is only touched inside
        // the critical section, since other threads may already be merging.
        const Extent empty;
        double lo0 = empty.lo[0], lo1 = empty.lo[1], lo2 = empty.lo[2];
        double hi0 = empty.hi[0], hi1 = empty.hi[1], hi2 = empty.hi[2];
        bool finite = true;
#if defined(_OPENMP)
#pragma omp for schedule(static) nowait
#endif
        for (int64_t i = 0; i < n; ++i) {
            const double x = values[i * 3];
            const double y = values[i * 3 + 1];
            const double z = values[i * 3 + 2];
            finite = finite & std::isfinite(x) & std::isfinite(y) & std::isfinite(z);
            lo0 = std::min(lo0, x);
            lo1 = std::min(lo1, y);
            lo2 = std::min(lo2, z);
            hi0 = std::max(hi0, x);
            hi1 = std::max(hi1, y);
            hi2 = std::max(hi2, z);
        }
#if defined(_OPENMP)
#pragma omp critical(hnw_asterism_extent)
#endif
        {
            extent->lo[0] = std::min(extent->lo[0], lo0);
            extent->lo[1] = std::min(extent->lo[1], lo1);
            extent->lo[2] = std::min(extent->lo[2], lo2);
            extent->hi[0] = std::max(extent->hi[0], hi0);
            extent->hi[1] = std::max(extent->hi[1], hi1);
            extent->hi[2] = std::max(extent->hi[2], hi2);
            extent->finite = extent->finite && finite;
        }
    }
}

int32_t token_cell(const double* point, const GridFrame& frame) {
    int64_t coords[3];
    for (int axis = 0; axis < 3; ++axis) {
        coords[axis] = std::min<int64_t>(
            frame.dims[axis] - 1,
            std::max<int64_t>(0,
                              token_cell_coordinate(point[axis], frame.origin[axis], frame.cell)));
    }
    return static_cast<int32_t>((coords[2] * frame.dims[1] + coords[1]) * frame.dims[0] +
                                coords[0]);
}

// Nearest point of `grid` to `query` among the 27 surrounding cells; -1 unless
// it lies within the threshold, kAmbiguousNearest when several points share
// that distance. Every point within the threshold is in those cells, so this
// equals the unbounded nearest neighbour whenever the latter is within it.
int32_t bounded_nearest(const double* query, const TokenGrid& grid, const GridFrame& frame,
                        const double threshold) {
    int64_t lo[3];
    int64_t hi[3];
    for (int axis = 0; axis < 3; ++axis) {
        const int64_t q = token_cell_coordinate(query[axis], frame.origin[axis], frame.cell);
        lo[axis] = std::max<int64_t>(0, q - 1);
        hi[axis] = std::min<int64_t>(frame.dims[axis] - 1, q + 1);
    }
    int32_t best = -1;
    bool tied = false;
    double best_distance = std::numeric_limits<double>::infinity();
    for (int64_t z = lo[2]; z <= hi[2]; ++z) {
        for (int64_t y = lo[1]; y <= hi[1]; ++y) {
            const int64_t row = (z * frame.dims[1] + y) * frame.dims[0];
            const int32_t begin = grid.cell_start[static_cast<size_t>(row + lo[0])];
            const int32_t end = grid.cell_start[static_cast<size_t>(row + hi[0] + 1)];
            for (int32_t slot = begin; slot < end; ++slot) {
                const int32_t j = grid.points[static_cast<size_t>(slot)];
                const double distance = token_squared_distance(
                    query, grid.coords.data() + static_cast<size_t>(slot) * 3);
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
    if (best < 0 || !(std::sqrt(best_distance) <= threshold)) {
        return -1;
    }
    return tied ? kAmbiguousNearest : best;
}

// Queries run in the query set's own cell order so consecutive queries probe
// the same neighbourhood of `grid` and hit cache; results land at the
// queries' original indices. Dense clusters make the per-query cost uneven,
// hence the dynamic schedule.
std::vector<int32_t> bounded_nearest_all(const TokenGrid& queries, const TokenGrid& grid,
                                         const GridFrame& frame, const double threshold) {
    const int64_t n_queries = static_cast<int64_t>(queries.points.size());
    std::vector<int32_t> nearest(static_cast<size_t>(n_queries));
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 256)
#endif
    for (int64_t slot = 0; slot < n_queries; ++slot) {
        nearest[static_cast<size_t>(queries.points[static_cast<size_t>(slot)])] = bounded_nearest(
            queries.coords.data() + static_cast<size_t>(slot) * 3, grid, frame, threshold);
    }
    return nearest;
}

} // namespace

void validate_inputs(const TokenArray& values1, const TokenArray& values2, const double threshold) {
    if (values1.ndim() != 2 || values1.shape(1) != 3 || values2.ndim() != 2 ||
        values2.shape(1) != 3) {
        throw std::invalid_argument("asterism_mutual_nearest: values must have shape (N, 3)");
    }
    if (!(threshold > 0.0) || !std::isfinite(threshold)) {
        throw std::invalid_argument("asterism_mutual_nearest: threshold must be positive");
    }
    if (values1.shape(0) > std::numeric_limits<int32_t>::max() ||
        values2.shape(0) > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument("asterism_mutual_nearest: too many tokens");
    }
}

int64_t token_cell_coordinate(const double value, const double origin, const double cell) {
    return static_cast<int64_t>(std::floor((value - origin) / cell));
}

bool finish_token_grid_frame(const double lo[3], const double hi[3], const double threshold,
                             GridFrame* frame) {
    frame->cell = threshold * (1.0 + kCellMargin);
    // Sized in floating point first: a huge extent must not reach the
    // integer conversion.
    double cells = 1.0;
    for (int axis = 0; axis < 3; ++axis) {
        const double span = std::floor((hi[axis] - lo[axis]) / frame->cell);
        cells *= span + 1.0;
        if (!(span >= 0.0) || !(cells <= static_cast<double>(kMaxGridCells))) {
            return false;
        }
        frame->origin[axis] = lo[axis];
        frame->dims[axis] = static_cast<int64_t>(span) + 1;
    }
    return true;
}

bool token_grid_frame(const double* values1, const int64_t n1, const double* values2,
                      const int64_t n2, const double threshold, GridFrame* frame) {
    Extent extent;
    accumulate_extent(values1, n1, &extent);
    accumulate_extent(values2, n2, &extent);
    return extent.finite && finish_token_grid_frame(extent.lo, extent.hi, threshold, frame);
}

TokenGrid build_token_grid(const double* values, const int64_t n, const GridFrame& frame) {
    const int64_t cell_count = frame.dims[0] * frame.dims[1] * frame.dims[2];
    std::vector<int32_t> point_cell(static_cast<size_t>(n));
    const int threads = light_pass_threads(n);
    (void)threads;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int64_t i = 0; i < n; ++i) {
        point_cell[static_cast<size_t>(i)] = token_cell(values + i * 3, frame);
    }
    // cell_start[c + 1] holds the count of cell c, then its start; the
    // scatter advances it to the start of cell c + 1, leaving CSR offsets.
    TokenGrid grid;
    grid.cell_start.assign(static_cast<size_t>(cell_count + 1), 0);
    for (int64_t i = 0; i < n; ++i) {
        ++grid.cell_start[static_cast<size_t>(point_cell[static_cast<size_t>(i)]) + 1];
    }
    int32_t running = 0;
    for (int64_t cell = 1; cell <= cell_count; ++cell) {
        const int32_t count = grid.cell_start[static_cast<size_t>(cell)];
        grid.cell_start[static_cast<size_t>(cell)] = running;
        running += count;
    }
    grid.points.resize(static_cast<size_t>(n));
    grid.coords.resize(static_cast<size_t>(n) * 3);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t slot =
            grid.cell_start[static_cast<size_t>(point_cell[static_cast<size_t>(i)]) + 1]++;
        grid.points[static_cast<size_t>(slot)] = static_cast<int32_t>(i);
        double* coords = grid.coords.data() + static_cast<size_t>(slot) * 3;
        coords[0] = values[i * 3];
        coords[1] = values[i * 3 + 1];
        coords[2] = values[i * 3 + 2];
    }
    return grid;
}

py::tuple mutual_pairs_to_arrays(const std::vector<int32_t>& mutual12) {
    const auto count =
        std::count_if(mutual12.begin(), mutual12.end(), [](const int32_t j) { return j >= 0; });
    py::array_t<int64_t> first(static_cast<ssize_t>(count));
    py::array_t<int64_t> second(static_cast<ssize_t>(count));
    auto* first_ptr = first.mutable_data();
    auto* second_ptr = second.mutable_data();
    size_t k = 0;
    for (size_t i = 0; i < mutual12.size(); ++i) {
        if (mutual12[i] >= 0) {
            first_ptr[k] = static_cast<int64_t>(i);
            second_ptr[k] = mutual12[i];
            ++k;
        }
    }
    return py::make_tuple(first, second);
}

bool has_ambiguous_nearest(const std::vector<int32_t>& nearest) {
    return std::find(nearest.begin(), nearest.end(), kAmbiguousNearest) != nearest.end();
}

py::tuple empty_pairs() {
    return mutual_pairs_to_arrays({});
}

} // namespace hnw::asterism

namespace {

using hnw::asterism::TokenArray;

py::object asterism_mutual_nearest_cpu_impl(const TokenArray& values1, const TokenArray& values2,
                                            const double threshold) {
    hnw::asterism::validate_inputs(values1, values2, threshold);
    const int64_t n1 = values1.shape(0);
    const int64_t n2 = values2.shape(0);
    if (n1 == 0 || n2 == 0) {
        return hnw::asterism::empty_pairs();
    }

    std::vector<int32_t> mutual12;
    bool representable = true;
    {
        py::gil_scoped_release release;
        hnw::asterism::GridFrame frame;
        representable = hnw::asterism::token_grid_frame(values1.data(), n1, values2.data(), n2,
                                                        threshold, &frame);
        if (representable) {
            const auto grid1 = hnw::asterism::build_token_grid(values1.data(), n1, frame);
            const auto grid2 = hnw::asterism::build_token_grid(values2.data(), n2, frame);
            mutual12 = hnw::asterism::bounded_nearest_all(grid1, grid2, frame, threshold);
            const auto nearest21 =
                hnw::asterism::bounded_nearest_all(grid2, grid1, frame, threshold);
            representable = !hnw::asterism::has_ambiguous_nearest(mutual12) &&
                            !hnw::asterism::has_ambiguous_nearest(nearest21);
            for (size_t i = 0; i < mutual12.size(); ++i) {
                const int32_t j = mutual12[i];
                if (j >= 0 && nearest21[static_cast<size_t>(j)] != static_cast<int32_t>(i)) {
                    mutual12[i] = -1;
                }
            }
        }
    }
    if (!representable) {
        return py::none();
    }
    return hnw::asterism::mutual_pairs_to_arrays(mutual12);
}

} // namespace

void bind_asterism_mutual_nearest_cpu_ops(py::module_& m) {
    m.def("asterism_mutual_nearest_cpu", &asterism_mutual_nearest_cpu_impl, py::arg("values1"),
          py::arg("values2"), py::arg("threshold"),
          "Mutual nearest asterism-token pairs within a Euclidean threshold using OpenMP; "
          "returns None when the token extent cannot be gridded or a nearest distance ties.");
}
