#include "alignment_ops.h"

#include "common/compat.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>

namespace {

#if defined(_MSC_VER)
#define HNW_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define HNW_RESTRICT __restrict__
#else
#define HNW_RESTRICT
#endif

constexpr ssize_t FEATURE_BINS = 120;
constexpr double FEATURE_STEP = 3.14159265358979323846 / 60.0;
constexpr int MIN_FILTERED_UNIQUE_PAIRS = 4;
constexpr int LOW_PAIR_COUNT_THRESHOLD = 10;
constexpr double MIN_FILTER_KEEP_RATIO = 0.5;

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

double cosine_distance(
    const double* HNW_RESTRICT a,
    const double* HNW_RESTRICT b,
    const ssize_t dim) {
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (ssize_t idx = 0; idx < dim; ++idx) {
        dot += a[idx] * b[idx];
        norm_a += a[idx] * a[idx];
        norm_b += b[idx] * b[idx];
    }
    if (norm_a == 0.0 || norm_b == 0.0) {
        return 1.0;
    }
    return 1.0 - dot / std::sqrt(norm_a * norm_b);
}

double percentile_linear(std::vector<double> values, const double percentile) {
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(values.begin(), values.end());
    if (values.size() == 1) {
        return values[0];
    }
    const double pos = (percentile / 100.0) * static_cast<double>(values.size() - 1);
    const auto lower_idx = static_cast<size_t>(std::floor(pos));
    const auto upper_idx = static_cast<size_t>(std::ceil(pos));
    const double weight = pos - static_cast<double>(lower_idx);
    return values[lower_idx] * (1.0 - weight) + values[upper_idx] * weight;
}

void inner_with_cross_matrix(
    const double* HNW_RESTRICT v,
    const double* HNW_RESTRICT base,
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

py::array_t<double> extract_point_features_impl(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& vec,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& vol,
    const int k) {
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
        return py::array_t<double>(std::vector<ssize_t>{0, FEATURE_BINS});
    }
    const ssize_t neighbor_count = std::min<ssize_t>(2 * static_cast<ssize_t>(k), n_points);
    if (neighbor_count < k) {
        throw std::invalid_argument("extract_point_features: k exceeds available neighbor count");
    }

    py::array_t<double> out(std::vector<ssize_t>{n_points, FEATURE_BINS});
    const auto vec_info = vec.request();
    const auto vol_info = vol.request();
    auto out_info = out.request();
    const auto* HNW_RESTRICT vec_ptr = static_cast<const double*>(vec_info.ptr);
    const auto* HNW_RESTRICT vol_ptr = static_cast<const double*>(vol_info.ptr);
    auto* HNW_RESTRICT out_ptr = static_cast<double*>(out_info.ptr);

    {
        py::gil_scoped_release release;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (ssize_t i = 0; i < n_points; ++i) {
            std::vector<ssize_t> order(static_cast<size_t>(n_points));
            std::iota(order.begin(), order.end(), 0);
            std::vector<double> similarities(static_cast<size_t>(n_points));
            const double* v0 = vec_ptr + i * 3;
            for (ssize_t j = 0; j < n_points; ++j) {
                similarities[static_cast<size_t>(j)] =
                    clamp_unit(cosine_similarity3(v0, vec_ptr + j * 3));
            }
            std::stable_sort(order.begin(), order.end(), [&](ssize_t lhs, ssize_t rhs) {
                return similarities[static_cast<size_t>(lhs)] >
                       similarities[static_cast<size_t>(rhs)];
            });

            std::vector<double> rho_pool(static_cast<size_t>(neighbor_count));
            std::vector<ssize_t> local_order(static_cast<size_t>(neighbor_count));
            std::iota(local_order.begin(), local_order.end(), 0);
            for (ssize_t j = 0; j < neighbor_count; ++j) {
                rho_pool[static_cast<size_t>(j)] =
                    std::acos(clamp_unit(similarities[static_cast<size_t>(order[j])]));
            }
            std::stable_sort(local_order.begin(), local_order.end(), [&](ssize_t lhs, ssize_t rhs) {
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
                const double s_norm =
                    std::sqrt(cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]);
                const double sign_dot = cr[0] * v0[0] + cr[1] * v0[1] + cr[2] * v0[2];
                const double s = s_norm * ((sign_dot > 0.0) - (sign_dot < 0.0));
                const double c = angle[0] * angle0[0] + angle[1] * angle0[1] + angle[2] * angle0[2];
                theta[static_cast<size_t>(jj)] = std::atan2(s, c);
                rho[static_cast<size_t>(jj)] = rho_pool[static_cast<size_t>(pool_pos)];
                selected_vol[static_cast<size_t>(jj)] = vol_ptr[src_idx];
            }

            double* out_row = out_ptr + i * FEATURE_BINS;
            std::fill(out_row, out_row + FEATURE_BINS, 0.0);
            for (int jj = 0; jj < k; ++jj) {
                const double sigma = 2.5 * std::exp(-rho[static_cast<size_t>(jj)] * 100.0) + 0.04;
                const double scale = selected_vol[static_cast<size_t>(jj)] *
                                     rho[static_cast<size_t>(jj)] *
                                     rho[static_cast<size_t>(jj)] / sigma;
                for (ssize_t bin = 0; bin < FEATURE_BINS; ++bin) {
                    const double fx = -3.14159265358979323846 + static_cast<double>(bin) * FEATURE_STEP;
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

    return out;
}

py::array_t<int32_t> find_initial_match_impl(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& features1,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& features2,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& pts1,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& pts2,
    const py::object& vectors1_obj,
    const py::object& vectors2_obj,
    const double alpha,
    const bool apply_threshold_filter,
    double theta_th,
    const double dist_multiplier) {
    if (features1.ndim() != 2 || features2.ndim() != 2 ||
        features1.shape(1) != features2.shape(1)) {
        throw std::invalid_argument(
            "find_initial_match: features must have shape (N, D) and (M, D)");
    }
    if (pts1.ndim() != 2 || pts1.shape(1) != 2 || pts1.shape(0) != features1.shape(0) ||
        pts2.ndim() != 2 || pts2.shape(1) != 2 || pts2.shape(0) != features2.shape(0)) {
        throw std::invalid_argument("find_initial_match: pts must have shape (N, 2)/(M, 2)");
    }

    py::array_t<double, py::array::c_style | py::array::forcecast> vectors1;
    py::array_t<double, py::array::c_style | py::array::forcecast> vectors2;
    if (apply_threshold_filter) {
        if (vectors1_obj.is_none() || vectors2_obj.is_none()) {
            throw std::invalid_argument(
                "find_initial_match: vectors1/vectors2 are required when threshold filter is enabled");
        }
        vectors1 = vectors1_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        vectors2 = vectors2_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        if (vectors1.ndim() != 2 || vectors1.shape(1) != 3 || vectors1.shape(0) != features1.shape(0) ||
            vectors2.ndim() != 2 || vectors2.shape(1) != 3 || vectors2.shape(0) != features2.shape(0)) {
            throw std::invalid_argument(
                "find_initial_match: vectors must have shape (N, 3)/(M, 3)");
        }
    }

    const ssize_t n1 = features1.shape(0);
    const ssize_t n2 = features2.shape(0);
    const ssize_t dim = features1.shape(1);
    if (n1 == 0 || n2 == 0) {
        return py::array_t<int32_t>(std::vector<ssize_t>{0, 2});
    }

    const auto f1_info = features1.request();
    const auto f2_info = features2.request();
    const auto pts1_info = pts1.request();
    const auto pts2_info = pts2.request();
    const auto* HNW_RESTRICT f1 = static_cast<const double*>(f1_info.ptr);
    const auto* HNW_RESTRICT f2 = static_cast<const double*>(f2_info.ptr);
    const auto* HNW_RESTRICT p1 = static_cast<const double*>(pts1_info.ptr);
    const auto* HNW_RESTRICT p2 = static_cast<const double*>(pts2_info.ptr);

    const double* HNW_RESTRICT v1 = nullptr;
    const double* HNW_RESTRICT v2 = nullptr;
    py::buffer_info vectors1_info;
    py::buffer_info vectors2_info;
    if (apply_threshold_filter) {
        vectors1_info = vectors1.request();
        vectors2_info = vectors2.request();
        v1 = static_cast<const double*>(vectors1_info.ptr);
        v2 = static_cast<const double*>(vectors2_info.ptr);
    }

    double pts_mean[2] = {0.0, 0.0};
    double pts_min[2] = {
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
    };
    double pts_max[2] = {
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };
    if (alpha > 0.0) {
        for (ssize_t i = 0; i < n1; ++i) {
            pts_mean[0] += p1[i * 2];
            pts_mean[1] += p1[i * 2 + 1];
            pts_min[0] = std::min(pts_min[0], p1[i * 2]);
            pts_min[1] = std::min(pts_min[1], p1[i * 2 + 1]);
            pts_max[0] = std::max(pts_max[0], p1[i * 2]);
            pts_max[1] = std::max(pts_max[1], p1[i * 2 + 1]);
        }
        for (ssize_t i = 0; i < n2; ++i) {
            pts_mean[0] += p2[i * 2];
            pts_mean[1] += p2[i * 2 + 1];
            pts_min[0] = std::min(pts_min[0], p2[i * 2]);
            pts_min[1] = std::min(pts_min[1], p2[i * 2 + 1]);
            pts_max[0] = std::max(pts_max[0], p2[i * 2]);
            pts_max[1] = std::max(pts_max[1], p2[i * 2 + 1]);
        }
        pts_mean[0] /= static_cast<double>(n1 + n2);
        pts_mean[1] /= static_cast<double>(n1 + n2);
    }

    std::vector<ssize_t> best12_idx(static_cast<size_t>(n1), 0);
    std::vector<ssize_t> best21_idx(static_cast<size_t>(n2), 0);
    std::vector<double> best12_dist(static_cast<size_t>(n1), std::numeric_limits<double>::infinity());
    std::vector<double> best21_dist(static_cast<size_t>(n2), std::numeric_limits<double>::infinity());
    std::vector<std::pair<int32_t, int32_t>> pairs;

    {
        py::gil_scoped_release release;

#if defined(_OPENMP)
#pragma omp parallel
    {
        std::vector<double> local_best21_dist(static_cast<size_t>(n2), std::numeric_limits<double>::infinity());
        std::vector<ssize_t> local_best21_idx(static_cast<size_t>(n2), 0);
#pragma omp for schedule(static)
        for (ssize_t i = 0; i < n1; ++i) {
            double row_best = std::numeric_limits<double>::infinity();
            ssize_t row_best_idx = 0;
            for (ssize_t j = 0; j < n2; ++j) {
                double dist = cosine_distance(f1 + i * dim, f2 + j * dim, dim);
                if (alpha > 0.0) {
                    const double dx = (p1[i * 2] - pts_mean[0]) /
                                          (pts_max[0] - pts_min[0]) -
                                      (p2[j * 2] - pts_mean[0]) /
                                          (pts_max[0] - pts_min[0]);
                    const double dy = (p1[i * 2 + 1] - pts_mean[1]) /
                                          (pts_max[1] - pts_min[1]) -
                                      (p2[j * 2 + 1] - pts_mean[1]) /
                                          (pts_max[1] - pts_min[1]);
                    dist = dist * (1.0 - alpha) + std::sqrt(dx * dx + dy * dy) * alpha;
                }
                if (dist < row_best) {
                    row_best = dist;
                    row_best_idx = j;
                }
                if (dist < local_best21_dist[static_cast<size_t>(j)]) {
                    local_best21_dist[static_cast<size_t>(j)] = dist;
                    local_best21_idx[static_cast<size_t>(j)] = i;
                }
            }
            best12_dist[static_cast<size_t>(i)] = row_best;
            best12_idx[static_cast<size_t>(i)] = row_best_idx;
        }
#pragma omp critical
        {
            for (ssize_t j = 0; j < n2; ++j) {
                if (local_best21_dist[static_cast<size_t>(j)] < best21_dist[static_cast<size_t>(j)]) {
                    best21_dist[static_cast<size_t>(j)] = local_best21_dist[static_cast<size_t>(j)];
                    best21_idx[static_cast<size_t>(j)] = local_best21_idx[static_cast<size_t>(j)];
                }
            }
        }
    }
#else
    for (ssize_t i = 0; i < n1; ++i) {
        for (ssize_t j = 0; j < n2; ++j) {
            double dist = cosine_distance(f1 + i * dim, f2 + j * dim, dim);
            if (alpha > 0.0) {
                const double dx = (p1[i * 2] - pts_mean[0]) /
                                      (pts_max[0] - pts_min[0]) -
                                  (p2[j * 2] - pts_mean[0]) /
                                      (pts_max[0] - pts_min[0]);
                const double dy = (p1[i * 2 + 1] - pts_mean[1]) /
                                      (pts_max[1] - pts_min[1]) -
                                  (p2[j * 2 + 1] - pts_mean[1]) /
                                      (pts_max[1] - pts_min[1]);
                dist = dist * (1.0 - alpha) + std::sqrt(dx * dx + dy * dy) * alpha;
            }
            if (dist < best12_dist[static_cast<size_t>(i)]) {
                best12_dist[static_cast<size_t>(i)] = dist;
                best12_idx[static_cast<size_t>(i)] = j;
            }
            if (dist < best21_dist[static_cast<size_t>(j)]) {
                best21_dist[static_cast<size_t>(j)] = dist;
                best21_idx[static_cast<size_t>(j)] = i;
            }
        }
    }
#endif

    const double d_th = std::min(percentile_linear(best12_dist, 30.0),
                                 percentile_linear(best21_dist, 30.0));
    for (ssize_t i = 0; i < n1; ++i) {
        const ssize_t j = best12_idx[static_cast<size_t>(i)];
        if (best21_idx[static_cast<size_t>(j)] == i &&
            best12_dist[static_cast<size_t>(i)] < d_th) {
            pairs.emplace_back(static_cast<int32_t>(i), static_cast<int32_t>(j));
        }
    }

    if (apply_threshold_filter && !pairs.empty()) {
        std::vector<std::pair<int32_t, int32_t>> unfiltered = pairs;
        std::vector<double> theta_values;
        theta_values.reserve(pairs.size());
        for (const auto& pair : pairs) {
            theta_values.push_back(std::acos(clamp_unit(
                v1[pair.first * 3] * v2[pair.second * 3] +
                v1[pair.first * 3 + 1] * v2[pair.second * 3 + 1] +
                v1[pair.first * 3 + 2] * v2[pair.second * 3 + 2])));
        }
        theta_th = std::min(percentile_linear(theta_values, 75.0), theta_th);

        double max_coord = 0.0;
        for (ssize_t i = 0; i < n1 * 2; ++i) {
            max_coord = std::max(max_coord, p1[i]);
        }
        for (ssize_t i = 0; i < n2 * 2; ++i) {
            max_coord = std::max(max_coord, p2[i]);
        }
        const double dist_th = max_coord * dist_multiplier;

        std::vector<std::pair<int32_t, int32_t>> filtered;
        filtered.reserve(pairs.size());
        for (size_t idx = 0; idx < pairs.size(); ++idx) {
            const auto& pair = pairs[idx];
            const double dx = p1[pair.first * 2] - p2[pair.second * 2];
            const double dy = p1[pair.first * 2 + 1] - p2[pair.second * 2 + 1];
            const double pts_dist = std::sqrt(dx * dx + dy * dy);
            if (theta_values[idx] < theta_th && pts_dist < dist_th) {
                filtered.push_back(pair);
            }
        }

        const double kept_ratio =
            unfiltered.empty() ? 0.0 : static_cast<double>(filtered.size()) /
                                      static_cast<double>(unfiltered.size());
        const bool fallback =
            filtered.size() < MIN_FILTERED_UNIQUE_PAIRS ||
            (unfiltered.size() < LOW_PAIR_COUNT_THRESHOLD && kept_ratio < MIN_FILTER_KEEP_RATIO);
        pairs = fallback ? std::move(unfiltered) : std::move(filtered);
    }
    }

    py::array_t<int32_t> out(
        std::vector<ssize_t>{static_cast<ssize_t>(pairs.size()), static_cast<ssize_t>(2)});
    auto out_info = out.request();
    auto* out_ptr = static_cast<int32_t*>(out_info.ptr);
    for (size_t idx = 0; idx < pairs.size(); ++idx) {
        out_ptr[idx * 2] = pairs[idx].first;
        out_ptr[idx * 2 + 1] = pairs[idx].second;
    }
    return out;
}

}  // namespace

void bind_alignment_ops(py::module_& m) {
    m.def("extract_point_features",
          &extract_point_features_impl,
          py::arg("vec"),
          py::arg("vol"),
          py::arg("k") = 15,
          "Extract star-point geometric descriptors using an OpenMP CPU kernel.");
    m.def("find_initial_match",
          &find_initial_match_impl,
          py::arg("features1"),
          py::arg("features2"),
          py::arg("pts1"),
          py::arg("pts2"),
          py::arg("vectors1") = py::none(),
          py::arg("vectors2") = py::none(),
          py::arg("alpha") = 0.0,
          py::arg("apply_threshold_filter") = true,
          py::arg("theta_th") = 3.14159265358979323846 / 6.0,
          py::arg("dist_multiplier") = 0.3,
          "Find initial mutual star-point matches using an OpenMP CPU kernel.");
}
