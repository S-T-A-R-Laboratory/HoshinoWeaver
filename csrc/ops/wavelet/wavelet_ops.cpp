#include "wavelet_ops.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>

namespace {

constexpr ssize_t DB8_FILTER_LEN = 16;
constexpr ssize_t DB8_DWT_OFFSET = -14;
constexpr ssize_t DB8_IDWT_OFFSET = 14;

constexpr std::array<double, DB8_FILTER_LEN> DB8_DEC_LO = {
    -0.00011747678412476953,
    0.0006754494064505693,
    -0.00039174037337694705,
    -0.004870352993451574,
    0.008746094047405777,
    0.013981027917398282,
    -0.044088253930794755,
    -0.017369301001807547,
    0.12874742662047847,
    0.0004724845739132828,
    -0.2840155429615469,
    -0.015829105256349306,
    0.5853546836542067,
    0.6756307362972898,
    0.31287159091429995,
    0.05441584224310401,
};

constexpr std::array<double, DB8_FILTER_LEN> DB8_DEC_HI = {
    -0.05441584224310401,
    0.31287159091429995,
    -0.6756307362972898,
    0.5853546836542067,
    0.015829105256349306,
    -0.2840155429615469,
    -0.0004724845739132828,
    0.12874742662047847,
    0.017369301001807547,
    -0.044088253930794755,
    -0.013981027917398282,
    0.008746094047405777,
    0.004870352993451574,
    -0.00039174037337694705,
    -0.0006754494064505693,
    -0.00011747678412476953,
};

constexpr std::array<double, DB8_FILTER_LEN> DB8_REC_LO = {
    0.05441584224310401,
    0.31287159091429995,
    0.6756307362972898,
    0.5853546836542067,
    -0.015829105256349306,
    -0.2840155429615469,
    0.0004724845739132828,
    0.12874742662047847,
    -0.017369301001807547,
    -0.044088253930794755,
    0.013981027917398282,
    0.008746094047405777,
    -0.004870352993451574,
    -0.00039174037337694705,
    0.0006754494064505693,
    -0.00011747678412476953,
};

constexpr std::array<double, DB8_FILTER_LEN> DB8_REC_HI = {
    -0.00011747678412476953,
    -0.0006754494064505693,
    -0.00039174037337694705,
    0.004870352993451574,
    0.008746094047405777,
    -0.013981027917398282,
    -0.044088253930794755,
    0.017369301001807547,
    0.12874742662047847,
    -0.0004724845739132828,
    -0.2840155429615469,
    0.015829105256349306,
    0.5853546836542067,
    -0.6756307362972898,
    0.31287159091429995,
    -0.05441584224310401,
};

struct DetailLevel {
    ssize_t h = 0;
    ssize_t w = 0;
    std::vector<double> cH;
    std::vector<double> cV;
    std::vector<double> cD;
};

inline ssize_t dwt_len(const ssize_t n) {
    return (n + DB8_FILTER_LEN - 1) / 2;
}

inline ssize_t idwt_len(const ssize_t n) {
    return 2 * n - DB8_FILTER_LEN + 2;
}

inline ssize_t symmetric_index(ssize_t idx, const ssize_t n) {
    if (n <= 1) {
        return 0;
    }
    const ssize_t period = 2 * n;
    idx %= period;
    if (idx < 0) {
        idx += period;
    }
    if (idx < n) {
        return idx;
    }
    return period - 1 - idx;
}

void dwt2(const std::vector<double>& input,
          const ssize_t h,
          const ssize_t w,
          std::vector<double>* approx,
          DetailLevel* detail) {
    const ssize_t out_h = dwt_len(h);
    const ssize_t out_w = dwt_len(w);
    std::vector<double> row_lo(static_cast<size_t>(h * out_w));
    std::vector<double> row_hi(static_cast<size_t>(h * out_w));

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ssize_t y = 0; y < h; ++y) {
        for (ssize_t x = 0; x < out_w; ++x) {
            double lo = 0.0;
            double hi = 0.0;
            for (ssize_t j = 0; j < DB8_FILTER_LEN; ++j) {
                const ssize_t src_x =
                    symmetric_index(2 * x + j + DB8_DWT_OFFSET, w);
                const double value = input[static_cast<size_t>(y * w + src_x)];
                const ssize_t rev_j = DB8_FILTER_LEN - 1 - j;
                lo += DB8_DEC_LO[static_cast<size_t>(rev_j)] * value;
                hi += DB8_DEC_HI[static_cast<size_t>(rev_j)] * value;
            }
            row_lo[static_cast<size_t>(y * out_w + x)] = lo;
            row_hi[static_cast<size_t>(y * out_w + x)] = hi;
        }
    }

    approx->assign(static_cast<size_t>(out_h * out_w), 0.0);
    detail->h = out_h;
    detail->w = out_w;
    detail->cH.assign(static_cast<size_t>(out_h * out_w), 0.0);
    detail->cV.assign(static_cast<size_t>(out_h * out_w), 0.0);
    detail->cD.assign(static_cast<size_t>(out_h * out_w), 0.0);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ssize_t y = 0; y < out_h; ++y) {
        for (ssize_t x = 0; x < out_w; ++x) {
            double ll = 0.0;
            double hl = 0.0;
            double lh = 0.0;
            double hh = 0.0;
            for (ssize_t j = 0; j < DB8_FILTER_LEN; ++j) {
                const ssize_t src_y =
                    symmetric_index(2 * y + j + DB8_DWT_OFFSET, h);
                const ssize_t rev_j = DB8_FILTER_LEN - 1 - j;
                const double row_lo_value = row_lo[static_cast<size_t>(
                    src_y * out_w + x)];
                const double row_hi_value = row_hi[static_cast<size_t>(
                    src_y * out_w + x)];
                ll += DB8_DEC_LO[static_cast<size_t>(rev_j)] * row_lo_value;
                hl += DB8_DEC_HI[static_cast<size_t>(rev_j)] * row_lo_value;
                lh += DB8_DEC_LO[static_cast<size_t>(rev_j)] * row_hi_value;
                hh += DB8_DEC_HI[static_cast<size_t>(rev_j)] * row_hi_value;
            }
            const size_t offset = static_cast<size_t>(y * out_w + x);
            (*approx)[offset] = ll;
            detail->cH[offset] = hl;
            detail->cV[offset] = lh;
            detail->cD[offset] = hh;
        }
    }
}

void crop_to(std::vector<double>* data,
             ssize_t* h,
             ssize_t* w,
             const ssize_t target_h,
             const ssize_t target_w) {
    if (*h == target_h && *w == target_w) {
        return;
    }
    if (*h < target_h || *w < target_w) {
        throw std::runtime_error("wavelet_dec_rec_cpu: invalid reconstruction shape");
    }
    std::vector<double> cropped(static_cast<size_t>(target_h * target_w));
    for (ssize_t y = 0; y < target_h; ++y) {
        std::copy_n(
            data->begin() + static_cast<size_t>(y * (*w)),
            static_cast<size_t>(target_w),
            cropped.begin() + static_cast<size_t>(y * target_w));
    }
    *data = std::move(cropped);
    *h = target_h;
    *w = target_w;
}

std::vector<double> idwt2(const std::vector<double>& approx,
                          const DetailLevel& detail,
                          const bool zero_detail,
                          ssize_t* out_h_ptr,
                          ssize_t* out_w_ptr) {
    const ssize_t h = detail.h;
    const ssize_t w = detail.w;
    const ssize_t out_h = idwt_len(h);
    const ssize_t out_w = idwt_len(w);
    std::vector<double> col_lo(static_cast<size_t>(out_h * w), 0.0);
    std::vector<double> col_hi(static_cast<size_t>(out_h * w), 0.0);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ssize_t y = 0; y < out_h; ++y) {
        for (ssize_t x = 0; x < w; ++x) {
            double lo = 0.0;
            double hi = 0.0;
            for (ssize_t j = 0; j < DB8_FILTER_LEN; ++j) {
                const ssize_t t = y + DB8_IDWT_OFFSET - j;
                if ((t & 1) != 0) {
                    continue;
                }
                const ssize_t src_y = symmetric_index(t / 2, h);
                const size_t offset = static_cast<size_t>(src_y * w + x);
                const double cA = approx[offset];
                const double cH = zero_detail ? 0.0 : detail.cH[offset];
                const double cV = zero_detail ? 0.0 : detail.cV[offset];
                const double cD = zero_detail ? 0.0 : detail.cD[offset];
                lo += DB8_REC_LO[static_cast<size_t>(j)] * cA +
                      DB8_REC_HI[static_cast<size_t>(j)] * cH;
                hi += DB8_REC_LO[static_cast<size_t>(j)] * cV +
                      DB8_REC_HI[static_cast<size_t>(j)] * cD;
            }
            col_lo[static_cast<size_t>(y * w + x)] = lo;
            col_hi[static_cast<size_t>(y * w + x)] = hi;
        }
    }

    std::vector<double> output(static_cast<size_t>(out_h * out_w), 0.0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ssize_t y = 0; y < out_h; ++y) {
        for (ssize_t x = 0; x < out_w; ++x) {
            double value = 0.0;
            for (ssize_t j = 0; j < DB8_FILTER_LEN; ++j) {
                const ssize_t t = x + DB8_IDWT_OFFSET - j;
                if ((t & 1) != 0) {
                    continue;
                }
                const ssize_t src_x = symmetric_index(t / 2, w);
                const size_t offset = static_cast<size_t>(y * w + src_x);
                value += DB8_REC_LO[static_cast<size_t>(j)] * col_lo[offset] +
                         DB8_REC_HI[static_cast<size_t>(j)] * col_hi[offset];
            }
            output[static_cast<size_t>(y * out_w + x)] = value;
        }
    }

    *out_h_ptr = out_h;
    *out_w_ptr = out_w;
    return output;
}

py::array_t<double> wavelet_dec_rec_cpu_impl(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& image,
    const ssize_t level) {
    if (image.ndim() != 2) {
        throw std::invalid_argument("wavelet_dec_rec_cpu: image must be 2D");
    }
    if (image.shape(0) <= 0 || image.shape(1) <= 0) {
        throw std::invalid_argument(
            "wavelet_dec_rec_cpu: image height and width must be positive");
    }
    if (level <= 0) {
        throw std::invalid_argument("wavelet_dec_rec_cpu: invalid wavelet level");
    }

    ssize_t current_h = image.shape(0);
    ssize_t current_w = image.shape(1);
    const double* input = image.data();
    std::vector<double> current(
        input,
        input + static_cast<size_t>(current_h * current_w));
    std::vector<DetailLevel> details(static_cast<size_t>(level));

    for (ssize_t idx = 0; idx < level; ++idx) {
        std::vector<double> approx;
        dwt2(current, current_h, current_w, &approx,
             &details[static_cast<size_t>(idx)]);
        current = std::move(approx);
        current_h = details[static_cast<size_t>(idx)].h;
        current_w = details[static_cast<size_t>(idx)].w;
    }

    std::fill(current.begin(), current.end(), 0.0);
    for (ssize_t idx = level - 1; idx >= 0; --idx) {
        const DetailLevel& detail = details[static_cast<size_t>(idx)];
        crop_to(&current, &current_h, &current_w, detail.h, detail.w);
        const bool zero_detail = idx == 0;
        current = idwt2(current, detail, zero_detail, &current_h, &current_w);
    }

    py::array_t<double> output({current_h, current_w});
    auto out = output.mutable_unchecked<2>();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ssize_t y = 0; y < current_h; ++y) {
        for (ssize_t x = 0; x < current_w; ++x) {
            out(y, x) = current[static_cast<size_t>(y * current_w + x)];
        }
    }
    return output;
}

}  // namespace

void bind_wavelet_ops(py::module_& m) {
    m.def(
        "wavelet_dec_rec_cpu",
        &wavelet_dec_rec_cpu_impl,
        py::arg("image"),
        py::arg("level"));
}
