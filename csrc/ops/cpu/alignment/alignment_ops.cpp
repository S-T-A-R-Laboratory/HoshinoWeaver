#include "alignment_ops.h"

#include "common/cpu_compat.h"
#include "ops/cpu/alignment/direction_grid.h"

#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

constexpr ssize_t FEATURE_BINS = hnw::alignment::kFeatureBins;
constexpr double FEATURE_STEP = 3.14159265358979323846 / 60.0;

double clamp_unit(const double value) {
    return std::max(-1.0, std::min(1.0, value));
}

double vector_norm3(const double* vec) {
    return std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}

double cosine_similarity3(const double* a, const double* b) {
    const double norm_a = vector_norm3(a);
    const double norm_b = vector_norm3(b);
    if (norm_a == 0.0 || norm_b == 0.0) {
        return 0.0;
    }
    return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (norm_a * norm_b);
}

void inner_with_cross_matrix(const double* HNW_RESTRICT v, const double* HNW_RESTRICT base,
                             double* HNW_RESTRICT out) {
    // Matches Python's np.inner(v, make_cross_matrix(base)).
    out[0] = -v[1] * base[2] + v[2] * base[1];
    out[1] = v[0] * base[2] - v[2] * base[0];
    out[2] = -v[0] * base[1] + v[1] * base[0];
}

void normalize3(double* vec) {
    const double norm = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
    if (norm == 0.0 || !std::isfinite(norm)) {
        vec[0] = 0.0;
        vec[1] = 0.0;
        vec[2] = 0.0;
        return;
    }
    vec[0] /= norm;
    vec[1] /= norm;
    vec[2] /= norm;
}

py::array_t<double> extract_point_features_impl(const hnw::alignment::PointArray& vec,
                                                const hnw::alignment::PointArray& vol,
                                                const int k) {
    const ssize_t neighbor_count = hnw::alignment::point_feature_pool_size(vec, vol, k);
    if (neighbor_count == 0) {
        return py::array_t<double>(std::vector<ssize_t>{0, FEATURE_BINS});
    }
    const ssize_t n_points = vec.shape(0);

    py::array_t<double> out(std::vector<ssize_t>{n_points, FEATURE_BINS});
    const auto vec_info = vec.request();
    const auto vol_info = vol.request();
    auto out_info = out.request();
    const auto* HNW_RESTRICT vec_ptr = static_cast<const double*>(vec_info.ptr);
    const auto* HNW_RESTRICT vol_ptr = static_cast<const double*>(vol_info.ptr);
    auto* HNW_RESTRICT out_ptr = static_cast<double*>(out_info.ptr);

    {
        py::gil_scoped_release release;
        const hnw::alignment::DirectionGrid grid(vec_ptr, n_points, neighbor_count);

#if defined(_OPENMP)
#pragma omp parallel
#endif
        {
            // Per-thread scratch reused across points: the two N-sized
            // buffers exceed the allocator's mmap threshold, so per-point
            // construction would pay one mmap/munmap pair per allocation.
            std::vector<ssize_t> order(static_cast<size_t>(n_points));
            std::vector<double> similarities(static_cast<size_t>(n_points));
            std::vector<double> rho_pool(static_cast<size_t>(neighbor_count));
            std::vector<ssize_t> local_order(static_cast<size_t>(neighbor_count));
            std::vector<double> grid_distances;
            std::vector<double> grid_kth;

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (ssize_t i = 0; i < n_points; ++i) {
                ssize_t candidate_count = n_points;
                if (grid.usable()) {
                    candidate_count =
                        grid.collect(i, neighbor_count, order.data(), &grid_distances, &grid_kth);
                } else {
                    std::iota(order.begin(), order.end(), 0);
                }
                const double* v0 = vec_ptr + i * 3;
                for (ssize_t c = 0; c < candidate_count; ++c) {
                    const ssize_t j = order[static_cast<size_t>(c)];
                    similarities[static_cast<size_t>(j)] =
                        clamp_unit(cosine_similarity3(v0, vec_ptr + j * 3));
                }
                // Only the top neighbor_count entries of `order` are consumed
                // below. The index tie-break makes this a total order whose
                // sorted prefix is bit-identical to the previous full
                // stable_sort for equal similarities.
                std::partial_sort(order.begin(), order.begin() + neighbor_count,
                                  order.begin() + candidate_count, [&](ssize_t lhs, ssize_t rhs) {
                                      const double lhs_sim = similarities[static_cast<size_t>(lhs)];
                                      const double rhs_sim = similarities[static_cast<size_t>(rhs)];
                                      if (lhs_sim != rhs_sim) {
                                          return lhs_sim > rhs_sim;
                                      }
                                      return lhs < rhs;
                                  });

                std::iota(local_order.begin(), local_order.end(), 0);
                for (ssize_t j = 0; j < neighbor_count; ++j) {
                    rho_pool[static_cast<size_t>(j)] =
                        std::acos(clamp_unit(similarities[static_cast<size_t>(order[j])]));
                }
                std::stable_sort(local_order.begin(), local_order.end(),
                                 [&](ssize_t lhs, ssize_t rhs) {
                                     const ssize_t lhs_idx = order[static_cast<size_t>(lhs)];
                                     const ssize_t rhs_idx = order[static_cast<size_t>(rhs)];
                                     return vol_ptr[lhs_idx] * rho_pool[static_cast<size_t>(lhs)] >
                                            vol_ptr[rhs_idx] * rho_pool[static_cast<size_t>(rhs)];
                                 });

                double angle0[3] = {0.0, 0.0, 0.0};
                bool have_angle0 = false;
                std::vector<double> theta(static_cast<size_t>(k));
                std::vector<double> rho(static_cast<size_t>(k));
                std::vector<double> selected_vol(static_cast<size_t>(k));

                for (int jj = 0; jj < k; ++jj) {
                    const ssize_t pool_pos = local_order[static_cast<size_t>(jj)];
                    const ssize_t src_idx = order[static_cast<size_t>(pool_pos)];
                    const double* vs = vec_ptr + src_idx * 3;

                    double angle[3];
                    inner_with_cross_matrix(vs, v0, angle);
                    normalize3(angle);
                    if (!have_angle0) {
                        angle0[0] = angle[0];
                        angle0[1] = angle[1];
                        angle0[2] = angle[2];
                        have_angle0 = true;
                    }

                    double cr[3];
                    inner_with_cross_matrix(angle, angle0, cr);
                    const double s_norm = std::sqrt(cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]);
                    const double sign_dot = cr[0] * v0[0] + cr[1] * v0[1] + cr[2] * v0[2];
                    const double s = s_norm * ((sign_dot > 0.0) - (sign_dot < 0.0));
                    const double c =
                        angle[0] * angle0[0] + angle[1] * angle0[1] + angle[2] * angle0[2];
                    theta[static_cast<size_t>(jj)] = std::atan2(s, c);
                    rho[static_cast<size_t>(jj)] = rho_pool[static_cast<size_t>(pool_pos)];
                    selected_vol[static_cast<size_t>(jj)] = vol_ptr[src_idx];
                }

                double* out_row = out_ptr + i * FEATURE_BINS;
                std::fill(out_row, out_row + FEATURE_BINS, 0.0);
                for (int jj = 0; jj < k; ++jj) {
                    const double sigma =
                        2.5 * std::exp(-rho[static_cast<size_t>(jj)] * 100.0) + 0.04;
                    const double scale = selected_vol[static_cast<size_t>(jj)] *
                                         rho[static_cast<size_t>(jj)] *
                                         rho[static_cast<size_t>(jj)] / sigma;
                    for (ssize_t bin = 0; bin < FEATURE_BINS; ++bin) {
                        const double fx =
                            -3.14159265358979323846 + static_cast<double>(bin) * FEATURE_STEP;
                        const double delta = theta[static_cast<size_t>(jj)] - fx;
                        out_row[bin] += std::exp(-(delta * delta) / (2.0 * sigma * sigma)) * scale;
                    }
                }

                double norm = 0.0;
                for (ssize_t bin = 0; bin < FEATURE_BINS; ++bin) {
                    norm += out_row[bin] * out_row[bin];
                }
                norm = std::sqrt(norm);
                if (norm > 0.0 && std::isfinite(norm)) {
                    for (ssize_t bin = 0; bin < FEATURE_BINS; ++bin) {
                        out_row[bin] /= norm;
                    }
                }
            }
        }
    }

    return out;
}

} // namespace

namespace hnw::alignment {

ssize_t point_feature_pool_size(const PointArray& vec, const PointArray& vol, const int k) {
    if (vec.ndim() != 2 || vec.shape(1) != 3) {
        throw std::invalid_argument("extract_point_features: vec must have shape (N, 3)");
    }
    if (vol.ndim() != 1 || vol.shape(0) != vec.shape(0)) {
        throw std::invalid_argument("extract_point_features: vol must have shape (N,)");
    }
    if (k <= 0) {
        throw std::invalid_argument("extract_point_features: k must be positive");
    }
    const ssize_t n_points = vec.shape(0);
    if (n_points <= 0) {
        return 0;
    }
    const ssize_t neighbor_count = std::min<ssize_t>(2 * static_cast<ssize_t>(k), n_points);
    if (neighbor_count < k) {
        throw std::invalid_argument("extract_point_features: k exceeds available neighbor count");
    }
    return neighbor_count;
}

} // namespace hnw::alignment

void bind_alignment_ops(py::module_& m) {
    m.def("extract_point_features", &extract_point_features_impl, py::arg("vec"), py::arg("vol"),
          py::arg("k") = 15,
          "Extract star-point geometric descriptors using an OpenMP CPU kernel.");
}
