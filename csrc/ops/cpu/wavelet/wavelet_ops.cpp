#include "wavelet_ops.h"

#include <pybind11/numpy.h>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr ssize_t DB8_FILTER_LEN = 16;
constexpr ssize_t DB8_DWT_OFFSET = -14;
constexpr ssize_t DB8_IDWT_OFFSET = 14;

constexpr std::array<double, DB8_FILTER_LEN> DB8_DEC_LO = {
    -0.00011747678412476953, 0.0006754494064505693, -0.00039174037337694705, -0.004870352993451574,
    0.008746094047405777,    0.013981027917398282,  -0.044088253930794755,   -0.017369301001807547,
    0.12874742662047847,     0.0004724845739132828, -0.2840155429615469,     -0.015829105256349306,
    0.5853546836542067,      0.6756307362972898,    0.31287159091429995,     0.05441584224310401,
};

constexpr std::array<double, DB8_FILTER_LEN> DB8_DEC_HI = {
    -0.05441584224310401, 0.31287159091429995,     -0.6756307362972898,    0.5853546836542067,
    0.015829105256349306, -0.2840155429615469,     -0.0004724845739132828, 0.12874742662047847,
    0.017369301001807547, -0.044088253930794755,   -0.013981027917398282,  0.008746094047405777,
    0.004870352993451574, -0.00039174037337694705, -0.0006754494064505693, -0.00011747678412476953,
};

constexpr std::array<double, DB8_FILTER_LEN> DB8_REC_LO = {
    0.05441584224310401,   0.31287159091429995,     0.6756307362972898,    0.5853546836542067,
    -0.015829105256349306, -0.2840155429615469,     0.0004724845739132828, 0.12874742662047847,
    -0.017369301001807547, -0.044088253930794755,   0.013981027917398282,  0.008746094047405777,
    -0.004870352993451574, -0.00039174037337694705, 0.0006754494064505693, -0.00011747678412476953,
};

constexpr std::array<double, DB8_FILTER_LEN> DB8_REC_HI = {
    -0.00011747678412476953, -0.0006754494064505693, -0.00039174037337694705, 0.004870352993451574,
    0.008746094047405777,    -0.013981027917398282,  -0.044088253930794755,   0.017369301001807547,
    0.12874742662047847,     -0.0004724845739132828, -0.2840155429615469,     0.015829105256349306,
    0.5853546836542067,      -0.6756307362972898,    0.31287159091429995,     -0.05441584224310401,
};

using hnw::wavelet::Buffer;

struct DetailLevel {
    ssize_t h = 0;
    ssize_t w = 0;
    Buffer cH;
    Buffer cV;
    Buffer cD;
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

// Source index of every (output, tap) pair along one axis. Building the
// symmetric extension once per level keeps the modulo out of pixel loops.
std::vector<ssize_t> dwt_source_indices(const ssize_t n) {
    const ssize_t out = dwt_len(n);
    std::vector<ssize_t> indices(static_cast<size_t>(out * DB8_FILTER_LEN));
    for (ssize_t pos = 0; pos < out; ++pos) {
        for (ssize_t j = 0; j < DB8_FILTER_LEN; ++j) {
            indices[static_cast<size_t>(pos * DB8_FILTER_LEN + j)] =
                symmetric_index(2 * pos + j + DB8_DWT_OFFSET, n);
        }
    }
    return indices;
}

// Only taps with an even upsampled position contribute to an idwt output, and
// their parity equals the output parity: taps j = (pos & 1) + 2 * m, in order.
constexpr ssize_t IDWT_TAPS = DB8_FILTER_LEN / 2;

std::vector<ssize_t> idwt_source_indices(const ssize_t out, const ssize_t n) {
    std::vector<ssize_t> indices(static_cast<size_t>(out * IDWT_TAPS));
    for (ssize_t pos = 0; pos < out; ++pos) {
        for (ssize_t m = 0; m < IDWT_TAPS; ++m) {
            const ssize_t j = (pos & 1) + 2 * m;
            indices[static_cast<size_t>(pos * IDWT_TAPS + m)] =
                symmetric_index((pos + DB8_IDWT_OFFSET - j) / 2, n);
        }
    }
    return indices;
}

void dwt2(const double* input, const ssize_t h, const ssize_t w, Buffer* approx,
          DetailLevel* detail, const bool keep_detail) {
    const ssize_t out_h = dwt_len(h);
    const ssize_t out_w = dwt_len(w);
    const std::vector<ssize_t> src_x = dwt_source_indices(w);
    const std::vector<ssize_t> src_y = dwt_source_indices(h);
    // Outputs whose 16 taps lie inside the row read them directly.
    const ssize_t interior_begin = (-DB8_DWT_OFFSET + 1) / 2;
    const ssize_t interior_end = (w - DB8_FILTER_LEN - DB8_DWT_OFFSET) / 2 + 1;
    Buffer row_lo(static_cast<size_t>(h * out_w));
    Buffer row_hi(keep_detail ? static_cast<size_t>(h * out_w) : 0);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ssize_t y = 0; y < h; ++y) {
        const double* row = input + y * w;
        for (ssize_t x = 0; x < out_w; ++x) {
            double lo = 0.0;
            double hi = 0.0;
            const auto accumulate = [&](const ssize_t j, const double value) {
                const ssize_t rev_j = DB8_FILTER_LEN - 1 - j;
                lo += DB8_DEC_LO[static_cast<size_t>(rev_j)] * value;
                if (keep_detail) {
                    hi += DB8_DEC_HI[static_cast<size_t>(rev_j)] * value;
                }
            };
            if (x >= interior_begin && x < interior_end) {
                const double* direct = row + 2 * x + DB8_DWT_OFFSET;
                for (ssize_t j = 0; j < DB8_FILTER_LEN; ++j) {
                    accumulate(j, direct[j]);
                }
            } else {
                const ssize_t* taps = src_x.data() + x * DB8_FILTER_LEN;
                for (ssize_t j = 0; j < DB8_FILTER_LEN; ++j) {
                    accumulate(j, row[taps[j]]);
                }
            }
            row_lo[static_cast<size_t>(y * out_w + x)] = lo;
            if (keep_detail) {
                row_hi[static_cast<size_t>(y * out_w + x)] = hi;
            }
        }
    }

    const size_t out_size = static_cast<size_t>(out_h * out_w);
    approx->resize(out_size);
    detail->h = out_h;
    detail->w = out_w;
    if (keep_detail) {
        detail->cH.resize(out_size);
        detail->cV.resize(out_size);
        detail->cD.resize(out_size);
    }

    // Accumulate one source row per tap into the whole output row: each
    // element still sums its taps in order, but memory is read sequentially.
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ssize_t y = 0; y < out_h; ++y) {
        const ssize_t* taps = src_y.data() + y * DB8_FILTER_LEN;
        double* ll = approx->data() + y * out_w;
        double* hl = keep_detail ? detail->cH.data() + y * out_w : nullptr;
        double* lh = keep_detail ? detail->cV.data() + y * out_w : nullptr;
        double* hh = keep_detail ? detail->cD.data() + y * out_w : nullptr;
        std::fill(ll, ll + out_w, 0.0);
        if (keep_detail) {
            std::fill(hl, hl + out_w, 0.0);
            std::fill(lh, lh + out_w, 0.0);
            std::fill(hh, hh + out_w, 0.0);
        }
        for (ssize_t j = 0; j < DB8_FILTER_LEN; ++j) {
            const ssize_t rev_j = DB8_FILTER_LEN - 1 - j;
            const double dec_lo = DB8_DEC_LO[static_cast<size_t>(rev_j)];
            const double dec_hi = DB8_DEC_HI[static_cast<size_t>(rev_j)];
            const double* lo_src = row_lo.data() + taps[j] * out_w;
            for (ssize_t x = 0; x < out_w; ++x) {
                ll[x] += dec_lo * lo_src[x];
            }
            if (keep_detail) {
                const double* hi_src = row_hi.data() + taps[j] * out_w;
                for (ssize_t x = 0; x < out_w; ++x) {
                    hl[x] += dec_hi * lo_src[x];
                    lh[x] += dec_lo * hi_src[x];
                    hh[x] += dec_hi * hi_src[x];
                }
            }
        }
    }
}

void crop_to(Buffer* data, ssize_t* h, ssize_t* w, const ssize_t target_h, const ssize_t target_w) {
    if (*h == target_h && *w == target_w) {
        return;
    }
    if (*h < target_h || *w < target_w) {
        throw std::runtime_error("wavelet_dec_rec_cpu: invalid reconstruction shape");
    }
    Buffer cropped(static_cast<size_t>(target_h * target_w));
    for (ssize_t y = 0; y < target_h; ++y) {
        std::copy_n(data->begin() + static_cast<size_t>(y * (*w)), static_cast<size_t>(target_w),
                    cropped.begin() + static_cast<size_t>(y * target_w));
    }
    *data = std::move(cropped);
    *h = target_h;
    *w = target_w;
}

// zero_detail marks the finest level, whose detail bands were never kept: its
// vertical/diagonal column pass is identically zero and is skipped.
Buffer idwt2(const Buffer& approx, const DetailLevel& detail, const bool zero_detail,
             ssize_t* out_h_ptr, ssize_t* out_w_ptr) {
    const ssize_t h = detail.h;
    const ssize_t w = detail.w;
    const ssize_t out_h = idwt_len(h);
    const ssize_t out_w = idwt_len(w);
    const std::vector<ssize_t> src_y = idwt_source_indices(out_h, h);
    const std::vector<ssize_t> src_x = idwt_source_indices(out_w, w);
    Buffer col_lo(static_cast<size_t>(out_h * w));
    Buffer col_hi(zero_detail ? 0 : static_cast<size_t>(out_h * w));

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ssize_t y = 0; y < out_h; ++y) {
        const ssize_t* taps = src_y.data() + y * IDWT_TAPS;
        const ssize_t first_tap = y & 1;
        double* lo = col_lo.data() + y * w;
        double* hi = zero_detail ? nullptr : col_hi.data() + y * w;
        std::fill(lo, lo + w, 0.0);
        if (!zero_detail) {
            std::fill(hi, hi + w, 0.0);
        }
        for (ssize_t m = 0; m < IDWT_TAPS; ++m) {
            const ssize_t j = first_tap + 2 * m;
            const double rec_lo = DB8_REC_LO[static_cast<size_t>(j)];
            const double rec_hi = DB8_REC_HI[static_cast<size_t>(j)];
            const size_t row_offset = static_cast<size_t>(taps[m] * w);
            const double* cA = approx.data() + row_offset;
            if (zero_detail) {
                const double cH = 0.0;
                for (ssize_t x = 0; x < w; ++x) {
                    lo[x] += rec_lo * cA[x] + rec_hi * cH;
                }
            } else {
                const double* cH = detail.cH.data() + row_offset;
                const double* cV = detail.cV.data() + row_offset;
                const double* cD = detail.cD.data() + row_offset;
                for (ssize_t x = 0; x < w; ++x) {
                    lo[x] += rec_lo * cA[x] + rec_hi * cH[x];
                    hi[x] += rec_lo * cV[x] + rec_hi * cD[x];
                }
            }
        }
    }

    // Output x reads source columns x / 2 + 7 - m for m = 0..7, all inside the
    // row (no symmetric extension) while x / 2 + 7 < w.
    const ssize_t interior_end = 2 * (w - IDWT_TAPS + 1);
    Buffer output(static_cast<size_t>(out_h * out_w));
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ssize_t y = 0; y < out_h; ++y) {
        const double* lo_row = col_lo.data() + y * w;
        const double* hi_row = zero_detail ? nullptr : col_hi.data() + y * w;
        for (ssize_t x = 0; x < out_w; ++x) {
            const ssize_t* taps = src_x.data() + x * IDWT_TAPS;
            const bool interior = x < interior_end;
            const ssize_t base = x / 2 + IDWT_TAPS - 1;
            const ssize_t first_tap = x & 1;
            double value = 0.0;
            for (ssize_t m = 0; m < IDWT_TAPS; ++m) {
                const ssize_t j = first_tap + 2 * m;
                const ssize_t source = interior ? base - m : taps[m];
                const double hi_value = zero_detail ? 0.0 : hi_row[source];
                value += DB8_REC_LO[static_cast<size_t>(j)] * lo_row[source] +
                         DB8_REC_HI[static_cast<size_t>(j)] * hi_value;
            }
            output[static_cast<size_t>(y * out_w + x)] = value;
        }
    }

    *out_h_ptr = out_h;
    *out_w_ptr = out_w;
    return output;
}

hnw::wavelet::CpuImage wavelet_dec_rec_cpu_core(const double* input, const ssize_t height,
                                                const ssize_t width, const ssize_t level) {
    ssize_t current_h = height;
    ssize_t current_w = width;
    Buffer current;
    std::vector<DetailLevel> details(static_cast<size_t>(level));

    for (ssize_t idx = 0; idx < level; ++idx) {
        Buffer approx;
        // The reconstruction below zeroes the finest detail band.
        dwt2(idx == 0 ? input : current.data(), current_h, current_w, &approx,
             &details[static_cast<size_t>(idx)], idx != 0);
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

    return {std::move(current), current_h, current_w};
}

py::array_t<double> wavelet_dec_rec_cpu_impl(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& image,
    const ssize_t level) {
    if (image.ndim() != 2) {
        throw std::invalid_argument("wavelet_dec_rec_cpu: image must be 2D");
    }
    if (image.shape(0) <= 0 || image.shape(1) <= 0) {
        throw std::invalid_argument("wavelet_dec_rec_cpu: image height and width must be positive");
    }
    if (level <= 0) {
        throw std::invalid_argument("wavelet_dec_rec_cpu: invalid wavelet level");
    }

    hnw::wavelet::CpuImage reconstructed;
    {
        py::gil_scoped_release release;
        reconstructed =
            hnw::wavelet::dec_rec_cpu(image.data(), image.shape(0), image.shape(1), level);
    }
    py::array_t<double> output({reconstructed.height, reconstructed.width});
    std::copy(reconstructed.values.begin(), reconstructed.values.end(), output.mutable_data());
    return output;
}

} // namespace

hnw::wavelet::CpuImage hnw::wavelet::dec_rec_cpu(const double* image, const int64_t height,
                                                 const int64_t width, const int64_t level) {
    if (image == nullptr) {
        throw std::invalid_argument("wavelet_dec_rec_cpu: image pointer must not be null");
    }
    if (height <= 0 || width <= 0) {
        throw std::invalid_argument("wavelet_dec_rec_cpu: image height and width must be positive");
    }
    if (level <= 0) {
        throw std::invalid_argument("wavelet_dec_rec_cpu: invalid wavelet level");
    }
    return wavelet_dec_rec_cpu_core(image, static_cast<ssize_t>(height),
                                    static_cast<ssize_t>(width), static_cast<ssize_t>(level));
}

void bind_wavelet_ops(py::module_& m) {
    m.def("wavelet_dec_rec_cpu", &wavelet_dec_rec_cpu_impl, py::arg("image"), py::arg("level"));
}
