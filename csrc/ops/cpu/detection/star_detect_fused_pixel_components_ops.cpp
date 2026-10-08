#include "star_detect_fused_pixel_components_ops.h"

#include "common/compat.h"
#include "common/cpu_compat.h"
#include "common/default_init_vector.h"
#include "common/wavelet_geometry.h"
#include "ops/cpu/wavelet/wavelet_ops.h"

#include <pybind11/numpy.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using hnw::wavelet::Buffer;

int64_t reflect101(int64_t index, const int64_t length) {
    if (length <= 1) {
        return 0;
    }
    while (index < 0 || index >= length) {
        index = index < 0 ? -index : 2 * length - index - 2;
    }
    return index;
}

void gaussian_rows(const double* input, double* output, const double* kernel, const int64_t height,
                   const int64_t width, const int64_t kernel_size) {
    const int64_t radius = kernel_size / 2;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t y = 0; y < height; ++y) {
        const double* row = input + y * width;
        for (int64_t x = 0; x < width; ++x) {
            double value = 0.0;
            if (x >= radius && x + radius < width) {
                const double* taps = row + x - radius;
                for (int64_t k = 0; k < kernel_size; ++k) {
                    value += taps[k] * kernel[k];
                }
            } else {
                for (int64_t k = 0; k < kernel_size; ++k) {
                    const int64_t xx = reflect101(x + k - radius, width);
                    value += row[xx] * kernel[k];
                }
            }
            output[y * width + x] = value;
        }
    }
}

void gaussian_cols(const double* input, double* output, const double* kernel, const int64_t height,
                   const int64_t width, const int64_t kernel_size) {
    const int64_t radius = kernel_size / 2;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t y = 0; y < height; ++y) {
        const bool interior = y >= radius && y + radius < height;
        for (int64_t x = 0; x < width; ++x) {
            double value = 0.0;
            if (interior) {
                const double* taps = input + (y - radius) * width + x;
                for (int64_t k = 0; k < kernel_size; ++k) {
                    value += taps[k * width] * kernel[k];
                }
            } else {
                for (int64_t k = 0; k < kernel_size; ++k) {
                    const int64_t yy = reflect101(y + k - radius, height);
                    value += input[yy * width + x] * kernel[k];
                }
            }
            output[y * width + x] = value;
        }
    }
}

void resize_linear(const double* input, double* output, const int64_t input_height,
                   const int64_t input_width, const int64_t output_height,
                   const int64_t output_width, const uint8_t* output_mask = nullptr) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t y = 0; y < output_height; ++y) {
        const double source_y = (static_cast<double>(y) + 0.5) * static_cast<double>(input_height) /
                                    static_cast<double>(output_height) -
                                0.5;
        const int64_t y0_raw = static_cast<int64_t>(std::floor(source_y));
        const double wy = source_y - static_cast<double>(y0_raw);
        const int64_t y0 = std::clamp<int64_t>(y0_raw, 0, input_height - 1);
        const int64_t y1 = std::clamp<int64_t>(y0_raw + 1, 0, input_height - 1);
        for (int64_t x = 0; x < output_width; ++x) {
            const int64_t offset = y * output_width + x;
            if (output_mask != nullptr && output_mask[offset] == 0) {
                output[offset] = 0.0;
                continue;
            }
            const double source_x = (static_cast<double>(x) + 0.5) *
                                        static_cast<double>(input_width) /
                                        static_cast<double>(output_width) -
                                    0.5;
            const int64_t x0_raw = static_cast<int64_t>(std::floor(source_x));
            const double wx = source_x - static_cast<double>(x0_raw);
            const int64_t x0 = std::clamp<int64_t>(x0_raw, 0, input_width - 1);
            const int64_t x1 = std::clamp<int64_t>(x0_raw + 1, 0, input_width - 1);
            const double top = input[y0 * input_width + x0] +
                               (input[y0 * input_width + x1] - input[y0 * input_width + x0]) * wx;
            const double bottom =
                input[y1 * input_width + x0] +
                (input[y1 * input_width + x1] - input[y1 * input_width + x0]) * wx;
            output[offset] = top + (bottom - top) * wy;
        }
    }
}

int max_team_size() {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int team_index() {
#if defined(_OPENMP)
    return omp_get_thread_num();
#else
    return 0;
#endif
}

// Order-preserving map from a double to an unsigned key (IEEE-754 order).
inline uint64_t order_key(const double value) {
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits >> 63) != 0 ? ~bits : bits | (uint64_t{1} << 63);
}

constexpr int SELECT_DIGIT_BITS = 16;
constexpr size_t SELECT_BINS = size_t{1} << SELECT_DIGIT_BITS;

// Masked-value counts per 16-bit key digit at `digit_shift`, limited to keys
// whose higher bits equal `prefix` (no limit for the top digit).
std::vector<uint64_t> masked_digit_histogram(const double* values, const uint8_t* mask,
                                             const int64_t total, const int digit_shift,
                                             const uint64_t prefix) {
    const bool has_prefix = digit_shift + SELECT_DIGIT_BITS < 64;
    const int prefix_shift = digit_shift + SELECT_DIGIT_BITS;
    std::vector<uint32_t> local(static_cast<size_t>(max_team_size()) * SELECT_BINS, 0);
#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        uint32_t* counts = local.data() + static_cast<size_t>(team_index()) * SELECT_BINS;
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int64_t index = 0; index < total; ++index) {
            if (mask[index] == 0) {
                continue;
            }
            const uint64_t key = order_key(values[index]);
            if (has_prefix && (key >> prefix_shift) != prefix) {
                continue;
            }
            ++counts[(key >> digit_shift) & (SELECT_BINS - 1)];
        }
    }
    std::vector<uint64_t> merged(SELECT_BINS, 0);
    for (size_t offset = 0; offset < local.size(); offset += SELECT_BINS) {
        for (size_t bin = 0; bin < SELECT_BINS; ++bin) {
            merged[bin] += local[offset + bin];
        }
    }
    return merged;
}

// Masked values whose key shares the top 32 bits located by two radix passes
// (the first pass histogram is supplied), with `rank` rebased into them.
std::vector<double> masked_rank_candidates(const double* values, const uint8_t* mask,
                                           const int64_t total,
                                           const std::vector<uint64_t>& top_histogram,
                                           uint64_t* rank) {
    uint64_t prefix = 0;
    std::vector<uint64_t> histogram = top_histogram;
    for (int digit_shift = 64 - SELECT_DIGIT_BITS; digit_shift >= 32;
         digit_shift -= SELECT_DIGIT_BITS) {
        if (digit_shift != 64 - SELECT_DIGIT_BITS) {
            histogram = masked_digit_histogram(values, mask, total, digit_shift, prefix);
        }
        size_t bin = 0;
        while (*rank >= histogram[bin]) {
            *rank -= histogram[bin];
            ++bin;
        }
        prefix = (prefix << SELECT_DIGIT_BITS) | bin;
    }

    std::vector<std::vector<double>> local(static_cast<size_t>(max_team_size()));
#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        std::vector<double>& part = local[static_cast<size_t>(team_index())];
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int64_t index = 0; index < total; ++index) {
            if (mask[index] != 0 && (order_key(values[index]) >> 32) == prefix) {
                part.push_back(values[index]);
            }
        }
    }
    std::vector<double> candidates;
    for (const std::vector<double>& part : local) {
        candidates.insert(candidates.end(), part.begin(), part.end());
    }
    return candidates;
}

// Smallest masked value strictly above `lower`, or `lower` itself when it
// repeats past sorted position `upper_index`.
double masked_next_value(const double* values, const uint8_t* mask, const int64_t total,
                         const double lower, const size_t upper_index) {
    int64_t not_above = 0;
    double above_min = std::numeric_limits<double>::infinity();
#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        int64_t local_not_above = 0;
        double local_above_min = std::numeric_limits<double>::infinity();
#if defined(_OPENMP)
#pragma omp for schedule(static) nowait
#endif
        for (int64_t index = 0; index < total; ++index) {
            if (mask[index] == 0) {
                continue;
            }
            const double value = values[index];
            if (lower < value) {
                local_above_min = std::min(local_above_min, value);
            } else {
                ++local_not_above;
            }
        }
#if defined(_OPENMP)
#pragma omp critical(hnw_star_detect_percentile)
#endif
        {
            not_above += local_not_above;
            above_min = std::min(above_min, local_above_min);
        }
    }
    return not_above > static_cast<int64_t>(upper_index) ? lower : above_min;
}

// np.percentile(values[mask], 99.5) with linear interpolation. The ranks are
// selected exactly (the values nth_element returns) without compacting the
// masked values.
double masked_percentile_995(const double* values, const uint8_t* mask, const int64_t total) {
    const std::vector<uint64_t> top_histogram =
        masked_digit_histogram(values, mask, total, 64 - SELECT_DIGIT_BITS, 0);
    uint64_t count = 0;
    for (const uint64_t bin_count : top_histogram) {
        count += bin_count;
    }
    if (count == 0) {
        throw std::invalid_argument("star_detect_fused_pixel_components: mask selects no pixels");
    }
    const double rank = 0.995 * static_cast<double>(count - 1);
    const size_t lower_index = static_cast<size_t>(std::floor(rank));
    const size_t upper_index = static_cast<size_t>(std::ceil(rank));

    uint64_t local_rank = lower_index;
    std::vector<double> candidates =
        masked_rank_candidates(values, mask, total, top_histogram, &local_rank);
    const auto lower_it = candidates.begin() + static_cast<ptrdiff_t>(local_rank);
    std::nth_element(candidates.begin(), lower_it, candidates.end());
    const double lower = *lower_it;
    if (lower_index == upper_index) {
        return lower;
    }
    double upper = 0.0;
    if (local_rank + 1 < candidates.size()) {
        const auto upper_it = lower_it + 1;
        std::nth_element(upper_it, upper_it, candidates.end());
        upper = *upper_it;
    } else {
        upper = masked_next_value(values, mask, total, lower, upper_index);
    }
    return lower + (upper - lower) * (rank - static_cast<double>(lower_index));
}

// Fixed-size blocks summed in index order and combined in block order, so the
// result does not depend on thread count or reduction arrival order.
double blocked_sum(const double* values, const int64_t total) {
    constexpr int64_t block = int64_t{1} << 16;
    const int64_t block_count = (total + block - 1) / block;
    std::vector<double> partial(static_cast<size_t>(block_count));
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t b = 0; b < block_count; ++b) {
        const int64_t end = std::min(total, (b + 1) * block);
        double sum = 0.0;
        for (int64_t index = b * block; index < end; ++index) {
            sum += values[index];
        }
        partial[static_cast<size_t>(b)] = sum;
    }
    double sum = 0.0;
    for (const double value : partial) {
        sum += value;
    }
    return sum;
}

void parallel_minmax(const double* values, const int64_t total, double* minimum, double* maximum) {
    double lo = values[0];
    double hi = values[0];
#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        double local_lo = values[0];
        double local_hi = values[0];
#if defined(_OPENMP)
#pragma omp for schedule(static) nowait
#endif
        for (int64_t index = 0; index < total; ++index) {
            local_lo = std::min(local_lo, values[index]);
            local_hi = std::max(local_hi, values[index]);
        }
#if defined(_OPENMP)
#pragma omp critical(hnw_star_detect_minmax)
#endif
        {
            lo = std::min(lo, local_lo);
            hi = std::max(hi, local_hi);
        }
    }
    *minimum = lo;
    *maximum = hi;
}

// One 3-tap AND (erode) or OR (dilate) along rows. Out-of-image neighbours
// pass an erosion and never set a dilation.
void morph_rows(const uint8_t* source, uint8_t* output, const int64_t height, const int64_t width,
                const bool erode) {
    const uint8_t outside = erode ? 1 : 0;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t y = 0; y < height; ++y) {
        const uint8_t* row = source + y * width;
        uint8_t* out = output + y * width;
        for (int64_t x = 0; x < width; ++x) {
            const uint8_t left = x > 0 ? row[x - 1] : outside;
            const uint8_t right = x + 1 < width ? row[x + 1] : outside;
            out[x] = erode ? (left & row[x] & right) : (left | row[x] | right);
        }
    }
}

// The matching 3-tap pass along columns; set pixels are written as `on`.
void morph_cols(const uint8_t* source, uint8_t* output, const int64_t height, const int64_t width,
                const bool erode, const uint8_t on) {
    const uint8_t outside = erode ? 1 : 0;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t y = 0; y < height; ++y) {
        const uint8_t* up = y > 0 ? source + (y - 1) * width : nullptr;
        const uint8_t* mid = source + y * width;
        const uint8_t* down = y + 1 < height ? source + (y + 1) * width : nullptr;
        uint8_t* out = output + y * width;
        for (int64_t x = 0; x < width; ++x) {
            const uint8_t above = up != nullptr ? up[x] : outside;
            const uint8_t below = down != nullptr ? down[x] : outside;
            const uint8_t value = erode ? (above & mid[x] & below) : (above | mid[x] | below);
            out[x] = value != 0 ? on : 0;
        }
    }
}

// 3x3 erosion then dilation, both clipped to the image; each square window is
// separable into a row and a column pass, which is exact for booleans.
void threshold_open(const double* image, const uint8_t* mask, uint8_t* output, const int64_t height,
                    const int64_t width, const double threshold) {
    const int64_t total = height * width;
    hnw::DefaultInitVector<uint8_t> binary(static_cast<size_t>(total));
    hnw::DefaultInitVector<uint8_t> scratch(static_cast<size_t>(total));
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t index = 0; index < total; ++index) {
        binary[static_cast<size_t>(index)] = mask[index] != 0 && image[index] > threshold ? 1 : 0;
    }
    morph_rows(binary.data(), scratch.data(), height, width, true);
    morph_cols(scratch.data(), binary.data(), height, width, true, 1);
    morph_rows(binary.data(), scratch.data(), height, width, false);
    morph_cols(scratch.data(), output, height, width, false, 255);
}

void connected_component_stats(const uint8_t* binary_mask, const double* image,
                               const int64_t height, const int64_t width,
                               std::vector<double>* positions, std::vector<double>* intensities) {
    const int64_t total = height * width;
    std::vector<uint8_t> visited(static_cast<size_t>(total), 0);
    std::vector<int64_t> queue;
    positions->clear();
    intensities->clear();

    for (int64_t start = 0; start < total; ++start) {
        if (binary_mask[start] == 0 || visited[static_cast<size_t>(start)] != 0) {
            continue;
        }
        queue.clear();
        queue.push_back(start);
        visited[static_cast<size_t>(start)] = 1;
        int64_t count = 0;
        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_intensity = 0.0;
        for (size_t head = 0; head < queue.size(); ++head) {
            const int64_t index = queue[head];
            const int64_t y = index / width;
            const int64_t x = index - y * width;
            ++count;
            sum_x += static_cast<double>(x);
            sum_y += static_cast<double>(y);
            sum_intensity += image[index];
            for (int64_t dy = -1; dy <= 1; ++dy) {
                const int64_t yy = y + dy;
                if (yy < 0 || yy >= height) {
                    continue;
                }
                for (int64_t dx = -1; dx <= 1; ++dx) {
                    const int64_t xx = x + dx;
                    if (xx < 0 || xx >= width) {
                        continue;
                    }
                    const int64_t neighbor = yy * width + xx;
                    if (binary_mask[neighbor] != 0 && visited[static_cast<size_t>(neighbor)] == 0) {
                        visited[static_cast<size_t>(neighbor)] = 1;
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        const double inv_count = 1.0 / static_cast<double>(count);
        positions->push_back(sum_x * inv_count);
        positions->push_back(sum_y * inv_count);
        intensities->push_back(sum_intensity * inv_count);
    }
}

void launch_star_detect_fused_pixel_components_cpu(
    const double* image, const uint8_t* external_mask, const double* gaussian_kernel,
    std::vector<double>* positions, std::vector<double>* intensities, uint8_t* binary_mask,
    const int64_t height, const int64_t width, const int64_t small_height,
    const int64_t small_width, const int64_t level, const int64_t gaussian_kernel_size) {
    const int64_t total = height * width;
    std::vector<uint8_t> full_mask;
    const uint8_t* mask = external_mask;
    if (mask == nullptr) {
        full_mask.assign(static_cast<size_t>(total), 1);
        mask = full_mask.data();
    }

    Buffer rows(static_cast<size_t>(total));
    Buffer normalized(static_cast<size_t>(total));
    gaussian_rows(image, rows.data(), gaussian_kernel, height, width, gaussian_kernel_size);
    gaussian_cols(rows.data(), normalized.data(), gaussian_kernel, height, width,
                  gaussian_kernel_size);
    Buffer().swap(rows);

    const double sum = blocked_sum(normalized.data(), total);
    double minimum = 0.0;
    double maximum = 0.0;
    parallel_minmax(normalized.data(), total, &minimum, &maximum);
    const double range = maximum - minimum;
    if (!(range > 0.0)) {
        throw std::runtime_error(
            "star_detect_fused_pixel_components: blurred image has zero range");
    }
    const double mean = sum / static_cast<double>(total);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t index = 0; index < total; ++index) {
        normalized[static_cast<size_t>(index)] =
            (normalized[static_cast<size_t>(index)] - mean) / range;
    }

    // Linear resizing between equal shapes samples every source pixel centre
    // with zero weight, so it is an exact copy and can be skipped.
    Buffer small;
    const double* wavelet_input = normalized.data();
    if (small_height != height || small_width != width) {
        small.resize(static_cast<size_t>(small_height * small_width));
        resize_linear(normalized.data(), small.data(), height, width, small_height, small_width);
        Buffer().swap(normalized);
        wavelet_input = small.data();
    }
    hnw::wavelet::CpuImage reconstructed =
        hnw::wavelet::dec_rec_cpu(wavelet_input, small_height, small_width, level);
    Buffer().swap(small);
    Buffer().swap(normalized);

    Buffer image_rec;
    if (reconstructed.height == height && reconstructed.width == width) {
        image_rec = std::move(reconstructed.values);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int64_t index = 0; index < total; ++index) {
            if (mask[index] == 0) {
                image_rec[static_cast<size_t>(index)] = 0.0;
            }
        }
    } else {
        image_rec.resize(static_cast<size_t>(total));
        resize_linear(reconstructed.values.data(), image_rec.data(), reconstructed.height,
                      reconstructed.width, height, width, mask);
        Buffer().swap(reconstructed.values);
    }

    const double threshold = masked_percentile_995(image_rec.data(), mask, total);
    threshold_open(image_rec.data(), mask, binary_mask, height, width, threshold);
    connected_component_stats(binary_mask, image_rec.data(), height, width, positions, intensities);
}

py::tuple star_detect_fused_pixel_components_cpu_impl(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& image,
    py::object mask_object, const ssize_t small_height, const ssize_t small_width,
    const ssize_t level,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& gaussian_kernel) {
    if (image.ndim() != 2) {
        throw std::invalid_argument("star_detect_fused_pixel_components: image must be 2D");
    }
    if (image.shape(0) <= 0 || image.shape(1) <= 0 || small_height <= 0 || small_width <= 0) {
        throw std::invalid_argument(
            "star_detect_fused_pixel_components: image dimensions must be positive");
    }
    if (image.shape(0) > std::numeric_limits<int>::max() / image.shape(1) ||
        small_height > std::numeric_limits<int>::max() / small_width) {
        throw std::invalid_argument("star_detect_fused_pixel_components: image is too large");
    }
    if (level <= 0) {
        throw std::invalid_argument("star_detect_fused_pixel_components: invalid wavelet level");
    }
    if (gaussian_kernel.ndim() != 1 || gaussian_kernel.shape(0) <= 0) {
        throw std::invalid_argument(
            "star_detect_fused_pixel_components: gaussian kernel must be 1D");
    }

    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask;
    const uint8_t* mask_pointer = nullptr;
    if (!mask_object.is_none()) {
        mask = mask_object.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
        if (mask.ndim() != 2 || mask.shape(0) != image.shape(0) ||
            mask.shape(1) != image.shape(1)) {
            throw std::invalid_argument(
                "star_detect_fused_pixel_components: mask shape must match image");
        }
        mask_pointer = mask.data();
    }

    const auto [output_height, output_width] =
        hnw::wavelet::reconstructed_shape(small_height, small_width, level);
    if (output_height <= 0 || output_width <= 0) {
        throw std::invalid_argument(
            "star_detect_fused_pixel_components: wavelet output shape is invalid");
    }

    std::vector<double> positions;
    std::vector<double> intensities;
    py::array_t<uint8_t> binary_mask({image.shape(0), image.shape(1)});
    {
        py::gil_scoped_release release;
        launch_star_detect_fused_pixel_components_cpu(
            image.data(), mask_pointer, gaussian_kernel.data(), &positions, &intensities,
            binary_mask.mutable_data(), image.shape(0), image.shape(1), small_height, small_width,
            level, gaussian_kernel.shape(0));
    }

    const ssize_t count = static_cast<ssize_t>(intensities.size());
    py::array_t<double> positions_output({count, static_cast<ssize_t>(2)});
    py::array_t<double> intensities_output({count});
    std::copy(positions.begin(), positions.end(), positions_output.mutable_data());
    std::copy(intensities.begin(), intensities.end(), intensities_output.mutable_data());
    return py::make_tuple(positions_output, intensities_output, binary_mask);
}

} // namespace

void bind_star_detect_fused_pixel_components_cpu_ops(py::module_& m) {
    m.def("star_detect_fused_pixel_components_cpu", &star_detect_fused_pixel_components_cpu_impl,
          py::arg("image"), py::arg("mask"), py::arg("small_height"), py::arg("small_width"),
          py::arg("level"), py::arg("gaussian_kernel"));
}
