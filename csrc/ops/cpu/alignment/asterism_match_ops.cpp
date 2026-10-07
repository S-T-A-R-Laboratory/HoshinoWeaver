#include "asterism_match_ops.h"

#include "asterism_mutual_nearest_ops.h"
#include "common/compat.h"
#include "common/cpu_compat.h"

#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

using VectorArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using TokenIndexArray = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using AnchorArray = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

constexpr double kMinLongEdge = 1e-12;
constexpr int64_t kParallelStars = 1024;
constexpr int kStarPassThreads = 8;
// Cell edge over the predicted ranked-th neighbour distance: most queries then
// stop after the first 3x3x3 block.
constexpr double kCellPerNeighbourDistance = 1.5;
constexpr double kGridMargin = 1e-7;

// Token arithmetic mirrors NumPy's left-to-right length-3 reductions.
// CMake disables FMA contraction for this file, including native/ARM builds.
double unit_dot(const double* a, const double* b) {
    const double p0 = a[0] * b[0];
    const double p1 = a[1] * b[1];
    const double p2 = a[2] * b[2];
    const double partial = p0 + p1;
    return partial + p2;
}

double unit_chord(const double dot) {
    const double clipped = std::min(std::max(dot, -1.0), 1.0);
    const double twice = 2.0 * clipped;
    const double chord2 = 2.0 - twice;
    return std::sqrt(std::max(chord2, 0.0));
}

std::vector<double> unit_vectors(const double* vectors, const int64_t n) {
    std::vector<double> unit(static_cast<size_t>(n) * 3);
    for (int64_t i = 0; i < n; ++i) {
        const double* v = vectors + i * 3;
        const double s0 = v[0] * v[0];
        const double s1 = v[1] * v[1];
        const double s2 = v[2] * v[2];
        const double partial = s0 + s1;
        const double norm = std::sqrt(partial + s2);
        for (int axis = 0; axis < 3; ++axis) {
            unit[static_cast<size_t>(i * 3 + axis)] = v[axis] / norm;
        }
    }
    return unit;
}

// Few-thread team for the per-star passes: they finish in milliseconds, and a
// full team on a many-core host costs more in fork/join and stalls than it saves.
int star_pass_threads(const int64_t n) {
#if defined(_OPENMP)
    return n >= kParallelStars ? std::min(omp_get_max_threads(), kStarPassThreads) : 1;
#else
    (void)n;
    return 1;
#endif
}

// Uniform grid over the unit vectors, CSR by cell, sized so the ranked-th
// nearest neighbour usually lies within one cell of the query.
struct NeighbourGrid {
    double origin[3] = {0.0, 0.0, 0.0};
    double cell = 0.0;
    int64_t dims[3] = {1, 1, 1};
    std::vector<int32_t> cell_start;
    std::vector<int32_t> points;

    int64_t coordinate(const double value, const int axis) const {
        const double offset = std::floor((value - origin[axis]) / cell);
        return std::min<int64_t>(dims[axis] - 1,
                                 std::max<int64_t>(0, static_cast<int64_t>(offset)));
    }
};

// Returns false for a degenerate extent; callers then scan every point.
bool build_neighbour_grid(const std::vector<double>& unit, const int64_t n, const int64_t ranked,
                          NeighbourGrid* grid) {
    double lo[3];
    double hi[3];
    for (int axis = 0; axis < 3; ++axis) {
        lo[axis] = hi[axis] = unit[static_cast<size_t>(axis)];
    }
    for (int64_t i = 1; i < n; ++i) {
        for (int axis = 0; axis < 3; ++axis) {
            const double value = unit[static_cast<size_t>(i * 3 + axis)];
            lo[axis] = std::min(lo[axis], value);
            hi[axis] = std::max(hi[axis], value);
        }
    }
    // One image's directions cover a surface patch: the density over the two
    // largest extents predicts the ranked-th neighbour distance.
    double extents[3] = {hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]};
    double sorted[3] = {extents[0], extents[1], extents[2]};
    std::sort(sorted, sorted + 3);
    const double area = sorted[2] * std::max(sorted[1], sorted[2] * 1e-3);
    if (!(area > 0.0)) {
        return false;
    }
    const double kPi = 3.14159265358979323846;
    grid->cell = kCellPerNeighbourDistance *
                 std::sqrt(area * static_cast<double>(ranked) / (kPi * static_cast<double>(n)));
    // A subnormal area can underflow during division; multiplying a zero
    // cell below would never grow it. Use the full scan for this extent.
    if (!(grid->cell > 0.0) || !std::isfinite(grid->cell)) {
        return false;
    }
    const double cell_limit = 8.0 * static_cast<double>(n) + 64.0;
    for (;;) {
        double cells = 1.0;
        for (int axis = 0; axis < 3; ++axis) {
            cells *= std::floor(extents[axis] / grid->cell) + 1.0;
        }
        if (cells <= cell_limit) {
            break;
        }
        grid->cell *= 1.25;
    }
    for (int axis = 0; axis < 3; ++axis) {
        grid->origin[axis] = lo[axis];
        grid->dims[axis] = static_cast<int64_t>(std::floor(extents[axis] / grid->cell)) + 1;
    }

    const int64_t cell_count = grid->dims[0] * grid->dims[1] * grid->dims[2];
    std::vector<int32_t> point_cell(static_cast<size_t>(n));
    grid->cell_start.assign(static_cast<size_t>(cell_count + 1), 0);
    for (int64_t i = 0; i < n; ++i) {
        const double* u = unit.data() + i * 3;
        const int64_t cell =
            (grid->coordinate(u[2], 2) * grid->dims[1] + grid->coordinate(u[1], 1)) *
                grid->dims[0] +
            grid->coordinate(u[0], 0);
        point_cell[static_cast<size_t>(i)] = static_cast<int32_t>(cell);
        ++grid->cell_start[static_cast<size_t>(cell + 1)];
    }
    for (int64_t cell = 0; cell < cell_count; ++cell) {
        grid->cell_start[static_cast<size_t>(cell + 1)] +=
            grid->cell_start[static_cast<size_t>(cell)];
    }
    std::vector<int32_t> fill(grid->cell_start.begin(), grid->cell_start.end() - 1);
    grid->points.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        grid->points[static_cast<size_t>(
            fill[static_cast<size_t>(point_cell[static_cast<size_t>(i)])]++)] =
            static_cast<int32_t>(i);
    }
    return true;
}

// The `ranked` smallest (squared distance, index) pairs seen, ascending.
class RankedNeighbours {
public:
    explicit RankedNeighbours(const int64_t ranked) : best_(static_cast<size_t>(ranked)) {}

    void clear() { count_ = 0; }
    bool full() const { return count_ == best_.size(); }
    double worst_distance() const { return best_[count_ - 1].first; }
    const std::pair<double, int32_t>& operator[](const size_t t) const { return best_[t]; }

    void offer(const double distance, const int32_t j) {
        const std::pair<double, int32_t> item(distance, j);
        if (full() && !(item < best_.back())) {
            return;
        }
        size_t t = full() ? count_ - 1 : count_++;
        for (; t > 0 && item < best_[t - 1]; --t) {
            best_[t] = best_[t - 1];
        }
        best_[t] = item;
    }

private:
    std::vector<std::pair<double, int32_t>> best_;
    size_t count_ = 0;
};

// Offers the points of cells x in [x0, x1] of one (y, z) row.
void offer_row(const NeighbourGrid& grid, const std::vector<double>& unit, const double* query,
               const int64_t row, const int64_t x0, const int64_t x1, RankedNeighbours* ranked) {
    const int32_t begin = grid.cell_start[static_cast<size_t>(row + x0)];
    const int32_t end = grid.cell_start[static_cast<size_t>(row + x1 + 1)];
    for (int32_t slot = begin; slot < end; ++slot) {
        const int32_t j = grid.points[static_cast<size_t>(slot)];
        ranked->offer(hnw::asterism::token_squared_distance(query, unit.data() + int64_t{j} * 3),
                      j);
    }
}

// Scans cell shells outward until the worst kept distance is below the
// distance to every unscanned cell by a margin far above rounding, so the
// kept set equals a full scan's, ties at its boundary included.
void grid_nearest(const NeighbourGrid& grid, const std::vector<double>& unit, const double* query,
                  RankedNeighbours* ranked) {
    int64_t center[3];
    for (int axis = 0; axis < 3; ++axis) {
        center[axis] = grid.coordinate(query[axis], axis);
    }
    for (int64_t r = 0;; ++r) {
        int64_t lo[3];
        int64_t hi[3];
        bool covers_grid = true;
        double gap = std::numeric_limits<double>::infinity();
        for (int axis = 0; axis < 3; ++axis) {
            lo[axis] = std::max<int64_t>(0, center[axis] - r);
            hi[axis] = std::min<int64_t>(grid.dims[axis] - 1, center[axis] + r);
            if (center[axis] - r > 0) {
                const double face =
                    grid.origin[axis] + static_cast<double>(center[axis] - r) * grid.cell;
                gap = std::min(gap, query[axis] - face);
                covers_grid = false;
            }
            if (center[axis] + r < grid.dims[axis] - 1) {
                const double face =
                    grid.origin[axis] + static_cast<double>(center[axis] + r + 1) * grid.cell;
                gap = std::min(gap, face - query[axis]);
                covers_grid = false;
            }
        }
        for (int64_t z = lo[2]; z <= hi[2]; ++z) {
            for (int64_t y = lo[1]; y <= hi[1]; ++y) {
                const int64_t row = (z * grid.dims[1] + y) * grid.dims[0];
                if (r == 0 || std::abs(z - center[2]) == r || std::abs(y - center[1]) == r) {
                    offer_row(grid, unit, query, row, lo[0], hi[0], ranked);
                    continue;
                }
                if (center[0] - r >= 0) {
                    offer_row(grid, unit, query, row, center[0] - r, center[0] - r, ranked);
                }
                if (center[0] + r < grid.dims[0]) {
                    offer_row(grid, unit, query, row, center[0] + r, center[0] + r, ranked);
                }
            }
        }
        if (covers_grid ||
            (ranked->full() && std::sqrt(ranked->worst_distance()) + kGridMargin < gap)) {
            return;
        }
    }
}

// The k nearest other points of every point, in SciPy cKDTree's order:
// ascending squared distance accumulated as the tree does. Returns false when
// two of the nearest k + 2 distances (the point itself, its k neighbours and
// the first excluded point) tie, since the tree's order there is unspecified.
bool nearest_neighbours(const std::vector<double>& unit, const int64_t n, const int64_t k,
                        std::vector<int32_t>* neighbours) {
    const int64_t ranked = std::min<int64_t>(k + 2, n);
    NeighbourGrid grid;
    const bool gridded = build_neighbour_grid(unit, n, ranked, &grid);
    neighbours->assign(static_cast<size_t>(n * k), 0);
    bool tied = false;
    const int threads = star_pass_threads(n);
    (void)threads;
#if defined(_OPENMP)
#pragma omp parallel num_threads(threads)
#endif
    {
        RankedNeighbours best(ranked);
        bool local_tied = false;
#if defined(_OPENMP)
#pragma omp for schedule(static) nowait
#endif
        for (int64_t i = 0; i < n; ++i) {
            if (local_tied) {
                continue;
            }
            const double* query = unit.data() + i * 3;
            best.clear();
            if (gridded) {
                grid_nearest(grid, unit, query, &best);
            } else {
                for (int64_t j = 0; j < n; ++j) {
                    best.offer(hnw::asterism::token_squared_distance(query, unit.data() + j * 3),
                               static_cast<int32_t>(j));
                }
            }
            local_tied = best[0].second != static_cast<int32_t>(i);
            for (int64_t t = 1; t < ranked; ++t) {
                local_tied = local_tied || best[static_cast<size_t>(t)].first ==
                                               best[static_cast<size_t>(t - 1)].first;
            }
            for (int64_t t = 1; t <= k; ++t) {
                (*neighbours)[static_cast<size_t>(i * k + t - 1)] =
                    best[static_cast<size_t>(t)].second;
            }
        }
#if defined(_OPENMP)
#pragma omp critical(hnw_asterism_tokens_tied)
#endif
        tied = tied || local_tied;
    }
    return !tied;
}

// Writes the tokens of anchor i to slots i * k(k-1)/2 onward, neighbour pairs
// in np.triu_indices(k, 1) order, with the long edge in place of its log;
// returns how many have a degenerate long edge.
int64_t anchor_tokens(const std::vector<double>& unit, const int64_t i, const int32_t* nb,
                      const int64_t k, double* anchor_edges, double* values, double* long_edge,
                      int32_t* anchors) {
    const double* anchor = unit.data() + i * 3;
    for (int64_t a = 0; a < k; ++a) {
        anchor_edges[a] = unit_chord(unit_dot(anchor, unit.data() + int64_t{nb[a]} * 3));
    }
    int64_t degenerate = 0;
    int64_t slot = i * (k * (k - 1) / 2);
    for (int64_t l = 0; l < k; ++l) {
        for (int64_t r = l + 1; r < k; ++r, ++slot) {
            const double first = anchor_edges[l];
            const double second = anchor_edges[r];
            const double longer = std::max(first, second);
            const double neighbour_edge = unit_chord(
                unit_dot(unit.data() + int64_t{nb[l]} * 3, unit.data() + int64_t{nb[r]} * 3));
            degenerate += longer > kMinLongEdge ? 0 : 1;
            values[slot * 3] = std::min(first, second) / longer;
            values[slot * 3 + 1] = neighbour_edge / longer;
            values[slot * 3 + 2] = longer;
            long_edge[slot] = longer;
            anchors[slot] = static_cast<int32_t>(i);
        }
    }
    return degenerate;
}

py::object asterism_tokens_cpu_impl(const VectorArray& vectors, const int64_t neighbor_count) {
    if (vectors.ndim() != 2 || vectors.shape(1) != 3) {
        throw std::invalid_argument("asterism_tokens: vectors must have shape (N, 3)");
    }
    const int64_t n = vectors.shape(0);
    if (n < 3 || neighbor_count < 2) {
        throw std::invalid_argument("asterism_tokens: need at least 3 vectors and 2 neighbours");
    }
    if (n > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument("asterism_tokens: too many vectors");
    }
    const int64_t k = std::min(neighbor_count, n - 1);
    const auto total = static_cast<ssize_t>(n * (k * (k - 1) / 2));

    py::array_t<double> values({total, static_cast<ssize_t>(3)});
    py::array_t<double> long_edge(total);
    py::array_t<int32_t> anchors(total);
    double* values_ptr = values.mutable_data();
    double* long_ptr = long_edge.mutable_data();
    int32_t* anchor_ptr = anchors.mutable_data();
    bool representable = true;
    int64_t degenerate = 0;
    {
        py::gil_scoped_release release;
        const std::vector<double> unit = unit_vectors(vectors.data(), n);
        std::vector<int32_t> neighbours;
        representable = nearest_neighbours(unit, n, k, &neighbours);
        if (representable) {
            const int threads = star_pass_threads(n);
            (void)threads;
#if defined(_OPENMP)
#pragma omp parallel num_threads(threads) reduction(+ : degenerate)
#endif
            {
                std::vector<double> anchor_edges(static_cast<size_t>(k));
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (int64_t i = 0; i < n; ++i) {
                    degenerate +=
                        anchor_tokens(unit, i, neighbours.data() + i * k, k, anchor_edges.data(),
                                      values_ptr, long_ptr, anchor_ptr);
                }
            }
        }
    }
    if (!representable) {
        return py::none();
    }
    if (degenerate == 0) {
        return py::make_tuple(values, long_edge, anchors);
    }

    // Coincident neighbours leave a zero long edge; drop those tokens as the
    // reference's boolean mask does.
    const auto count = total - static_cast<ssize_t>(degenerate);
    py::array_t<double> values_kept({count, static_cast<ssize_t>(3)});
    py::array_t<double> long_kept(count);
    py::array_t<int32_t> anchors_kept(count);
    double* values_out = values_kept.mutable_data();
    double* long_out = long_kept.mutable_data();
    int32_t* anchors_out = anchors_kept.mutable_data();
    ssize_t out = 0;
    for (ssize_t slot = 0; slot < total; ++slot) {
        if (long_ptr[slot] > kMinLongEdge) {
            std::copy(values_ptr + slot * 3, values_ptr + slot * 3 + 3, values_out + out * 3);
            long_out[out] = long_ptr[slot];
            anchors_out[out] = anchor_ptr[slot];
            ++out;
        }
    }
    return py::make_tuple(values_kept, long_kept, anchors_kept);
}

// One vote tally side: the top two vote counts seen so far and the partner
// that first reached the top count.
void record_vote(const int32_t partner, const int32_t votes, int32_t* best_partner,
                 int32_t* best_votes, int32_t* second_votes) {
    if (votes > *best_votes) {
        *second_votes = *best_votes;
        *best_votes = votes;
        *best_partner = partner;
    } else if (votes > *second_votes) {
        *second_votes = votes;
    }
}

std::vector<int32_t> gather_anchors(const TokenIndexArray& token_pairs, const AnchorArray& anchors,
                                    const int64_t anchor_count) {
    const int64_t n = token_pairs.shape(0);
    const int64_t tokens = anchors.shape(0);
    const int64_t* token_ptr = token_pairs.data();
    const int32_t* anchor_ptr = anchors.data();
    std::vector<int32_t> gathered(static_cast<size_t>(n));
    for (int64_t t = 0; t < n; ++t) {
        const int64_t token = token_ptr[t];
        if (token < 0 || token >= tokens) {
            throw std::invalid_argument("asterism_anchor_votes: token index out of range");
        }
        const int32_t anchor = anchor_ptr[token];
        if (anchor < 0 || anchor >= anchor_count) {
            throw std::invalid_argument("asterism_anchor_votes: anchor index out of range");
        }
        gathered[static_cast<size_t>(t)] = anchor;
    }
    return gathered;
}

py::tuple asterism_anchor_votes_cpu_impl(const TokenIndexArray& token_pairs1,
                                         const TokenIndexArray& token_pairs2,
                                         const AnchorArray& anchors1, const AnchorArray& anchors2,
                                         const int64_t num1, const int64_t num2,
                                         const int64_t min_votes, const int64_t min_vote_margin) {
    if (token_pairs1.ndim() != 1 || token_pairs2.ndim() != 1 || anchors1.ndim() != 1 ||
        anchors2.ndim() != 1 || token_pairs1.shape(0) != token_pairs2.shape(0)) {
        throw std::invalid_argument(
            "asterism_anchor_votes: token pairs and anchors must be matching 1-D arrays");
    }
    if (num1 < 0 || num2 < 0 || num1 > std::numeric_limits<int32_t>::max() ||
        num2 > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument("asterism_anchor_votes: invalid star counts");
    }
    const std::vector<int32_t> first = gather_anchors(token_pairs1, anchors1, num1);
    const std::vector<int32_t> second = gather_anchors(token_pairs2, anchors2, num2);
    const int64_t n = static_cast<int64_t>(first.size());

    std::vector<int32_t> accepted_pairs;
    std::vector<int32_t> accepted_votes;
    int64_t voted_pairs = 0;
    {
        py::gil_scoped_release release;
        // Bucket by first anchor and sort each bucket: (first, second) pairs
        // are then visited in the ascending pair-code order of np.unique.
        std::vector<int64_t> start(static_cast<size_t>(num1 + 1), 0);
        for (int64_t t = 0; t < n; ++t) {
            ++start[static_cast<size_t>(first[static_cast<size_t>(t)]) + 1];
        }
        for (int64_t a = 0; a < num1; ++a) {
            start[static_cast<size_t>(a + 1)] += start[static_cast<size_t>(a)];
        }
        std::vector<int64_t> fill(start.begin(), start.end() - 1);
        std::vector<int32_t> ordered(static_cast<size_t>(n));
        for (int64_t t = 0; t < n; ++t) {
            ordered[static_cast<size_t>(
                fill[static_cast<size_t>(first[static_cast<size_t>(t)])]++)] =
                second[static_cast<size_t>(t)];
        }

        std::vector<int32_t> best2_for_1(static_cast<size_t>(num1), -1);
        std::vector<int32_t> best_votes1(static_cast<size_t>(num1), 0);
        std::vector<int32_t> second_votes1(static_cast<size_t>(num1), 0);
        std::vector<int32_t> best1_for_2(static_cast<size_t>(num2), -1);
        std::vector<int32_t> best_votes2(static_cast<size_t>(num2), 0);
        std::vector<int32_t> second_votes2(static_cast<size_t>(num2), 0);
        for (int64_t a = 0; a < num1; ++a) {
            const auto begin = ordered.begin() + start[static_cast<size_t>(a)];
            const auto end = ordered.begin() + start[static_cast<size_t>(a + 1)];
            std::sort(begin, end);
            for (auto run = begin; run != end;) {
                const int32_t b = *run;
                const auto run_end = std::upper_bound(run, end, b);
                const auto votes = static_cast<int32_t>(run_end - run);
                ++voted_pairs;
                record_vote(b, votes, &best2_for_1[static_cast<size_t>(a)],
                            &best_votes1[static_cast<size_t>(a)],
                            &second_votes1[static_cast<size_t>(a)]);
                record_vote(static_cast<int32_t>(a), votes, &best1_for_2[static_cast<size_t>(b)],
                            &best_votes2[static_cast<size_t>(b)],
                            &second_votes2[static_cast<size_t>(b)]);
                run = run_end;
            }
        }

        const auto decisive = [&](const int32_t best, const int32_t runner_up) {
            return int64_t{best} >= min_votes && int64_t{best} - runner_up >= min_vote_margin;
        };
        for (int64_t a = 0; a < num1; ++a) {
            const int32_t b = best2_for_1[static_cast<size_t>(a)];
            if (b < 0 ||
                !decisive(best_votes1[static_cast<size_t>(a)],
                          second_votes1[static_cast<size_t>(a)]) ||
                best1_for_2[static_cast<size_t>(b)] != a ||
                !decisive(best_votes2[static_cast<size_t>(b)],
                          second_votes2[static_cast<size_t>(b)])) {
                continue;
            }
            accepted_pairs.push_back(static_cast<int32_t>(a));
            accepted_pairs.push_back(b);
            accepted_votes.push_back(best_votes1[static_cast<size_t>(a)]);
        }
    }

    const auto accepted = static_cast<ssize_t>(accepted_votes.size());
    py::array_t<int32_t> pair_idx({accepted, static_cast<ssize_t>(2)});
    py::array_t<int32_t> votes_out(accepted);
    std::copy(accepted_pairs.begin(), accepted_pairs.end(), pair_idx.mutable_data());
    std::copy(accepted_votes.begin(), accepted_votes.end(), votes_out.mutable_data());
    return py::make_tuple(pair_idx, voted_pairs, votes_out);
}

} // namespace

void bind_asterism_match_cpu_ops(py::module_& m) {
    m.def("asterism_tokens_cpu", &asterism_tokens_cpu_impl, py::arg("vectors"),
          py::arg("neighbor_count"),
          "Local spherical-triangle tokens using OpenMP as (values, long_edge, anchors), the "
          "long edge standing in for its log in column 2; None when a neighbour distance ties.");
    m.def("asterism_anchor_votes_cpu", &asterism_anchor_votes_cpu_impl, py::arg("token_pairs1"),
          py::arg("token_pairs2"), py::arg("anchors1"), py::arg("anchors2"), py::arg("num1"),
          py::arg("num2"), py::arg("min_votes"), py::arg("min_vote_margin"),
          "Mutual anchor pairs from token-pair votes: (pair_idx, voted_pairs, accepted_votes).");
}
