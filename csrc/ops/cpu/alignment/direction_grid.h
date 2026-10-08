#pragma once

#include "common/compat.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace hnw::alignment {

// Uniform grid over unit directions for candidate-neighbor generation.
// collect() visits cell shells outward until the neighbor_count-th nearest
// direction found is closer than any unvisited cell by a margin far above
// rounding error, so the exact cosine ranking that follows sees a superset of
// the true nearest set and selects exactly what a full scan would.
class DirectionGrid {
public:
    DirectionGrid(const double* vec, const ssize_t n_points, const ssize_t neighbor_count) {
        if (n_points < kMinGridPoints || neighbor_count >= n_points) {
            return;
        }
        unit_.resize(static_cast<size_t>(n_points * 3));
        double lo[3] = {0.0, 0.0, 0.0};
        double hi[3] = {0.0, 0.0, 0.0};
        for (ssize_t i = 0; i < n_points; ++i) {
            const double* v = vec + i * 3;
            const double norm2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            // Far from under/overflow, unit vectors and exact cosines stay
            // accurate well within kGridMargin; otherwise use the full scan.
            if (!(norm2 >= kMinNorm2 && norm2 <= kMaxNorm2)) {
                return;
            }
            const double norm = std::sqrt(norm2);
            for (int axis = 0; axis < 3; ++axis) {
                const double value = vec[i * 3 + axis] / norm;
                unit_[static_cast<size_t>(i * 3 + axis)] = value;
                lo[axis] = i == 0 ? value : std::min(lo[axis], value);
                hi[axis] = i == 0 ? value : std::max(hi[axis], value);
            }
        }
        double extents[3];
        for (int axis = 0; axis < 3; ++axis) {
            min_[axis] = lo[axis];
            extents[axis] = hi[axis] - lo[axis];
        }
        // Directions of one image cover a surface patch; size cells from the
        // two largest extents for a few points per occupied cell.
        double sorted[3] = {extents[0], extents[1], extents[2]};
        std::sort(sorted, sorted + 3);
        const double area = sorted[2] * std::max(sorted[1], sorted[2] * 1e-3);
        if (!(area > 0.0)) {
            return;
        }
        cell_ = std::sqrt(area / std::max<double>(1.0, static_cast<double>(n_points) / 4.0));
        const double cell_limit = 8.0 * static_cast<double>(n_points) + 64.0;
        for (;;) {
            double cells = 1.0;
            for (int axis = 0; axis < 3; ++axis) {
                dims_[axis] = static_cast<int64_t>(std::floor(extents[axis] / cell_)) + 1;
                cells *= static_cast<double>(dims_[axis]);
            }
            if (cells <= cell_limit) {
                break;
            }
            cell_ *= 1.25;
        }

        const int64_t cell_count = dims_[0] * dims_[1] * dims_[2];
        std::vector<int64_t> point_cell(static_cast<size_t>(n_points));
        cell_start_.assign(static_cast<size_t>(cell_count + 1), 0);
        for (ssize_t i = 0; i < n_points; ++i) {
            int64_t coords[3];
            point_cell[static_cast<size_t>(i)] = cell_of(i, coords);
            ++cell_start_[static_cast<size_t>(point_cell[static_cast<size_t>(i)] + 1)];
        }
        for (int64_t cell = 0; cell < cell_count; ++cell) {
            cell_start_[static_cast<size_t>(cell + 1)] += cell_start_[static_cast<size_t>(cell)];
        }
        std::vector<int64_t> fill(cell_start_.begin(), cell_start_.end() - 1);
        cell_points_.resize(static_cast<size_t>(n_points));
        for (ssize_t i = 0; i < n_points; ++i) {
            const int64_t cell = point_cell[static_cast<size_t>(i)];
            cell_points_[static_cast<size_t>(fill[static_cast<size_t>(cell)]++)] = i;
        }
        usable_ = true;
    }

    bool usable() const { return usable_; }
    const std::vector<double>& unit() const { return unit_; }
    const double* min() const { return min_; }
    double cell() const { return cell_; }
    const int64_t* dims() const { return dims_; }
    const std::vector<int64_t>& cell_start() const { return cell_start_; }
    const std::vector<ssize_t>& cell_points() const { return cell_points_; }

    static constexpr ssize_t kMinGridPoints = 512;
    static constexpr double kGridMargin = 1e-7;
    static constexpr double kMinNorm2 = 1e-270;
    static constexpr double kMaxNorm2 = 1e270;

    // Writes candidate indices to out (capacity n_points) and returns their count.
    ssize_t collect(const ssize_t i, const ssize_t neighbor_count, ssize_t* out,
                    std::vector<double>* distances, std::vector<double>* kth_scratch) const {
        int64_t q[3];
        cell_of(i, q);
        const double* u0 = unit_.data() + i * 3;
        distances->clear();
        ssize_t count = 0;
        for (int64_t r = 0;; ++r) {
            int64_t lo[3];
            int64_t hi[3];
            bool covers_grid = true;
            for (int axis = 0; axis < 3; ++axis) {
                lo[axis] = std::max<int64_t>(0, q[axis] - r);
                hi[axis] = std::min<int64_t>(dims_[axis] - 1, q[axis] + r);
                covers_grid = covers_grid && q[axis] - r <= 0 && q[axis] + r >= dims_[axis] - 1;
            }
            for (int64_t z = lo[2]; z <= hi[2]; ++z) {
                for (int64_t y = lo[1]; y <= hi[1]; ++y) {
                    for (int64_t x = lo[0]; x <= hi[0]; ++x) {
                        const int64_t ring =
                            std::max({std::abs(x - q[0]), std::abs(y - q[1]), std::abs(z - q[2])});
                        if (ring != r) {
                            continue;
                        }
                        const int64_t cell = (z * dims_[1] + y) * dims_[0] + x;
                        for (int64_t slot = cell_start_[static_cast<size_t>(cell)];
                             slot < cell_start_[static_cast<size_t>(cell + 1)]; ++slot) {
                            const ssize_t j = cell_points_[static_cast<size_t>(slot)];
                            const double* u1 = unit_.data() + j * 3;
                            const double dx = u0[0] - u1[0];
                            const double dy = u0[1] - u1[1];
                            const double dz = u0[2] - u1[2];
                            out[count++] = j;
                            distances->push_back(dx * dx + dy * dy + dz * dz);
                        }
                    }
                }
            }
            if (covers_grid) {
                return count;
            }
            if (count >= neighbor_count) {
                kth_scratch->assign(distances->begin(), distances->end());
                std::nth_element(kth_scratch->begin(), kth_scratch->begin() + (neighbor_count - 1),
                                 kth_scratch->end());
                // Unvisited cells lie farther than r cells from the query.
                const double kth =
                    std::sqrt((*kth_scratch)[static_cast<size_t>(neighbor_count - 1)]);
                if (kth + kGridMargin < static_cast<double>(r) * cell_) {
                    return count;
                }
            }
        }
    }

private:
    int64_t cell_of(const ssize_t i, int64_t coords[3]) const {
        for (int axis = 0; axis < 3; ++axis) {
            const double offset = (unit_[static_cast<size_t>(i * 3 + axis)] - min_[axis]) / cell_;
            coords[axis] = std::min<int64_t>(dims_[axis] - 1,
                                             std::max<int64_t>(0, static_cast<int64_t>(offset)));
        }
        return (coords[2] * dims_[1] + coords[1]) * dims_[0] + coords[0];
    }

    bool usable_ = false;
    std::vector<double> unit_;
    double min_[3] = {0.0, 0.0, 0.0};
    double cell_ = 0.0;
    int64_t dims_[3] = {1, 1, 1};
    std::vector<int64_t> cell_start_;
    std::vector<ssize_t> cell_points_;
};

} // namespace hnw::alignment
