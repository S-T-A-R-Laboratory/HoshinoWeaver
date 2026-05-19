#include "fgp_ops.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <pybind11/numpy.h>

namespace {

#if defined(_MSC_VER)
#define HNW_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define HNW_RESTRICT __restrict__
#else
#define HNW_RESTRICT
#endif

#ifndef HNW_ENABLE_OMP_SIMD
#define HNW_ENABLE_OMP_SIMD 0
#endif

void validate_accumulator_shapes(const py::array& sum_mu,
                                 const py::array& square_sum,
                                 const py::array& n,
                                 const py::array& fresh,
                                 const char* op_name) {
    if (sum_mu.ndim() != fresh.ndim() || square_sum.ndim() != fresh.ndim() ||
        n.ndim() != fresh.ndim()) {
        throw std::invalid_argument(std::string(op_name) + ": ndim mismatch");
    }
    for (ssize_t i = 0; i < fresh.ndim(); ++i) {
        if (sum_mu.shape(i) != fresh.shape(i) ||
            square_sum.shape(i) != fresh.shape(i) ||
            n.shape(i) != fresh.shape(i)) {
            throw std::invalid_argument(std::string(op_name) + ": shape mismatch");
        }
    }
}

template <typename FreshT, typename SumT, typename SquareT, typename CountT>
void fgp_accumulate_inplace_kernel(py::buffer_info& sum_info,
                                   py::buffer_info& square_info,
                                   py::buffer_info& count_info,
                                   const py::buffer_info& fresh_info,
                                   const uint64_t weight) {
    auto* HNW_RESTRICT sum_ptr = static_cast<SumT*>(sum_info.ptr);
    auto* HNW_RESTRICT square_ptr = static_cast<SquareT*>(square_info.ptr);
    auto* HNW_RESTRICT count_ptr = static_cast<CountT*>(count_info.ptr);
    const auto* HNW_RESTRICT fresh_ptr =
        static_cast<const FreshT*>(fresh_info.ptr);
    const ssize_t total = fresh_info.size;

    using SumAccumT = SumT;
    using SquareAccumT = SquareT;
    using CountAccumT = CountT;
    const SumAccumT weight_sum = static_cast<SumAccumT>(weight);
    const SquareAccumT weight_square = static_cast<SquareAccumT>(weight);
    const CountAccumT weight_count = static_cast<CountAccumT>(weight);

    py::gil_scoped_release release;
#if defined(_OPENMP) && HNW_ENABLE_OMP_SIMD
#pragma omp parallel for simd schedule(static)
#elif defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ssize_t i = 0; i < total; ++i) {
        const SumAccumT value_sum = static_cast<SumAccumT>(fresh_ptr[i]);
        const SquareAccumT value_square =
            static_cast<SquareAccumT>(fresh_ptr[i]);
        sum_ptr[i] = static_cast<SumT>(
            static_cast<SumAccumT>(sum_ptr[i]) +
            static_cast<SumAccumT>(value_sum * weight_sum));
        square_ptr[i] = static_cast<SquareT>(
            static_cast<SquareAccumT>(square_ptr[i]) +
            static_cast<SquareAccumT>(value_square * value_square * weight_square));
        count_ptr[i] = static_cast<CountT>(
            static_cast<CountAccumT>(count_ptr[i]) + weight_count);
    }
}

template <typename SumT, typename SquareT, typename CountT>
void fgp_add_inplace_kernel(py::buffer_info& sum_info,
                            py::buffer_info& square_info,
                            py::buffer_info& count_info,
                            const py::buffer_info& other_sum_info,
                            const py::buffer_info& other_square_info,
                            const py::buffer_info& other_count_info) {
    auto* HNW_RESTRICT sum_ptr = static_cast<SumT*>(sum_info.ptr);
    auto* HNW_RESTRICT square_ptr = static_cast<SquareT*>(square_info.ptr);
    auto* HNW_RESTRICT count_ptr = static_cast<CountT*>(count_info.ptr);
    const auto* HNW_RESTRICT other_sum_ptr =
        static_cast<const SumT*>(other_sum_info.ptr);
    const auto* HNW_RESTRICT other_square_ptr =
        static_cast<const SquareT*>(other_square_info.ptr);
    const auto* HNW_RESTRICT other_count_ptr =
        static_cast<const CountT*>(other_count_info.ptr);
    const ssize_t total = sum_info.size;

    py::gil_scoped_release release;
#if defined(_OPENMP) && HNW_ENABLE_OMP_SIMD
#pragma omp parallel for simd schedule(static)
#elif defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ssize_t i = 0; i < total; ++i) {
        sum_ptr[i] = static_cast<SumT>(sum_ptr[i] + other_sum_ptr[i]);
        square_ptr[i] =
            static_cast<SquareT>(square_ptr[i] + other_square_ptr[i]);
        count_ptr[i] = static_cast<CountT>(count_ptr[i] + other_count_ptr[i]);
    }
}

template <typename FreshT, typename SumT, typename SquareT, typename CountT>
void fgp_masked_mean_inplace_kernel(py::buffer_info& sum_info,
                                    py::buffer_info& square_info,
                                    py::buffer_info& count_info,
                                    const py::buffer_info& fresh_info,
                                    const py::buffer_info& mask_info) {
    auto* HNW_RESTRICT sum_ptr = static_cast<SumT*>(sum_info.ptr);
    auto* HNW_RESTRICT square_ptr = static_cast<SquareT*>(square_info.ptr);
    auto* HNW_RESTRICT count_ptr = static_cast<CountT*>(count_info.ptr);
    const auto* HNW_RESTRICT fresh_ptr =
        static_cast<const FreshT*>(fresh_info.ptr);
    const auto* HNW_RESTRICT mask_ptr = static_cast<const uint8_t*>(mask_info.ptr);
    const ssize_t spatial = mask_info.size;
    const ssize_t channels = spatial == 0 ? 0 : fresh_info.size / spatial;

    py::gil_scoped_release release;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ssize_t idx = 0; idx < spatial; ++idx) {
        if (mask_ptr[idx] == 0) {
            continue;
        }
        const ssize_t base = idx * channels;
#if defined(HNW_ENABLE_OMP_SIMD) && HNW_ENABLE_OMP_SIMD
#pragma omp simd
#endif
        for (ssize_t c = 0; c < channels; ++c) {
            const ssize_t offset = base + c;
            const SumT value_sum = static_cast<SumT>(fresh_ptr[offset]);
            const SquareT value_square = static_cast<SquareT>(fresh_ptr[offset]);
            sum_ptr[offset] = static_cast<SumT>(sum_ptr[offset] + value_sum);
            square_ptr[offset] = static_cast<SquareT>(
                square_ptr[offset] + value_square * value_square);
            count_ptr[offset] = static_cast<CountT>(count_ptr[offset] + 1);
        }
    }
}

template <typename FreshT, typename SumT, typename SquareT, typename CountT>
void sigma_clip_fused_inplace_kernel(py::buffer_info& sum_info,
                                     py::buffer_info& square_info,
                                     py::buffer_info& count_info,
                                     const py::buffer_info& fresh_info,
                                     const py::buffer_info& rej_high_info,
                                     const py::buffer_info& rej_low_info) {
    auto* HNW_RESTRICT sum_ptr = static_cast<SumT*>(sum_info.ptr);
    auto* HNW_RESTRICT square_ptr = static_cast<SquareT*>(square_info.ptr);
    auto* HNW_RESTRICT count_ptr = static_cast<CountT*>(count_info.ptr);
    const auto* HNW_RESTRICT fresh_ptr =
        static_cast<const FreshT*>(fresh_info.ptr);
    const auto* HNW_RESTRICT rej_high_ptr =
        static_cast<const FreshT*>(rej_high_info.ptr);
    const auto* HNW_RESTRICT rej_low_ptr =
        static_cast<const FreshT*>(rej_low_info.ptr);
    const ssize_t total = fresh_info.size;

    py::gil_scoped_release release;
#if defined(_OPENMP) && HNW_ENABLE_OMP_SIMD
#pragma omp parallel for simd schedule(static)
#elif defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ssize_t i = 0; i < total; ++i) {
        const FreshT value = fresh_ptr[i];
        if (value < rej_low_ptr[i] || value > rej_high_ptr[i]) {
            const SumT value_sum = static_cast<SumT>(value);
            const SquareT value_square = static_cast<SquareT>(value);
            sum_ptr[i] = static_cast<SumT>(sum_ptr[i] + value_sum);
            square_ptr[i] = static_cast<SquareT>(
                square_ptr[i] + value_square * value_square);
            count_ptr[i] = static_cast<CountT>(count_ptr[i] + 1);
        }
    }
}

template <typename FreshT, typename SumT, typename SquareT, typename CountT>
void sigma_clip_fused_masked_inplace_kernel(py::buffer_info& sum_info,
                                            py::buffer_info& square_info,
                                            py::buffer_info& count_info,
                                            const py::buffer_info& fresh_info,
                                            const py::buffer_info& rej_high_info,
                                            const py::buffer_info& rej_low_info,
                                            const py::buffer_info& mask_info) {
    auto* HNW_RESTRICT sum_ptr = static_cast<SumT*>(sum_info.ptr);
    auto* HNW_RESTRICT square_ptr = static_cast<SquareT*>(square_info.ptr);
    auto* HNW_RESTRICT count_ptr = static_cast<CountT*>(count_info.ptr);
    const auto* HNW_RESTRICT fresh_ptr =
        static_cast<const FreshT*>(fresh_info.ptr);
    const auto* HNW_RESTRICT rej_high_ptr =
        static_cast<const FreshT*>(rej_high_info.ptr);
    const auto* HNW_RESTRICT rej_low_ptr =
        static_cast<const FreshT*>(rej_low_info.ptr);
    const auto* HNW_RESTRICT mask_ptr = static_cast<const uint8_t*>(mask_info.ptr);
    const ssize_t spatial = mask_info.size;
    const ssize_t channels = spatial == 0 ? 0 : fresh_info.size / spatial;

    py::gil_scoped_release release;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ssize_t idx = 0; idx < spatial; ++idx) {
        if (mask_ptr[idx] == 0) {
            continue;
        }
        const ssize_t base = idx * channels;
#if defined(HNW_ENABLE_OMP_SIMD) && HNW_ENABLE_OMP_SIMD
#pragma omp simd
#endif
        for (ssize_t c = 0; c < channels; ++c) {
            const ssize_t offset = base + c;
            const FreshT value = fresh_ptr[offset];
            if (value < rej_low_ptr[offset] || value > rej_high_ptr[offset]) {
                const SumT value_sum = static_cast<SumT>(value);
                const SquareT value_square = static_cast<SquareT>(value);
                sum_ptr[offset] = static_cast<SumT>(sum_ptr[offset] + value_sum);
                square_ptr[offset] = static_cast<SquareT>(
                    square_ptr[offset] + value_square * value_square);
                count_ptr[offset] = static_cast<CountT>(count_ptr[offset] + 1);
            }
        }
    }
}

template <typename FreshT>
void huber_weighted_accumulate_inplace_kernel(
    py::buffer_info& weighted_sum_info,
    py::buffer_info& weight_total_info,
    const py::buffer_info& fresh_info,
    const py::buffer_info& ref_mean_info,
    const py::buffer_info& ref_std_info,
    const double huber_c,
    const double frame_weight) {
    auto* HNW_RESTRICT weighted_sum_ptr =
        static_cast<double*>(weighted_sum_info.ptr);
    auto* HNW_RESTRICT weight_total_ptr =
        static_cast<double*>(weight_total_info.ptr);
    const auto* HNW_RESTRICT fresh_ptr =
        static_cast<const FreshT*>(fresh_info.ptr);
    const auto* HNW_RESTRICT ref_mean_ptr =
        static_cast<const float*>(ref_mean_info.ptr);
    const auto* HNW_RESTRICT ref_std_ptr =
        static_cast<const float*>(ref_std_info.ptr);
    const ssize_t total = fresh_info.size;
    const float huber_c_f = static_cast<float>(huber_c);

    py::gil_scoped_release release;
#if defined(_OPENMP) && HNW_ENABLE_OMP_SIMD
#pragma omp parallel for simd schedule(static)
#elif defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (ssize_t i = 0; i < total; ++i) {
        const double pixel_value = static_cast<double>(fresh_ptr[i]);
        const float residual_value = static_cast<float>(fresh_ptr[i]);
        const float residual =
            (residual_value - ref_mean_ptr[i]) / (ref_std_ptr[i] + 1.0e-10f);
        const float abs_residual = std::fabs(residual);
        const float huber_weight = abs_residual <= huber_c_f
                                       ? 1.0f
                                       : huber_c_f / (abs_residual + 1.0e-10f);
        const double effective_weight =
            static_cast<double>(huber_weight) * frame_weight;
        weighted_sum_ptr[i] += pixel_value * effective_weight;
        weight_total_ptr[i] += effective_weight;
    }
}

template <typename FreshT, typename SumT, typename SquareT>
void dispatch_accumulate_count_dtype(
    py::array_t<SumT, py::array::c_style | py::array::forcecast> sum_mu,
    py::array_t<SquareT, py::array::c_style | py::array::forcecast> square_sum,
    py::array n,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& fresh,
    const uint64_t weight) {
    if (py::isinstance<py::array_t<uint16_t>>(n)) {
        auto count = n.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        fgp_accumulate_inplace_kernel<FreshT, SumT, SquareT, uint16_t>(
            sum_info, square_info, count_info, fresh_info, weight);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(n)) {
        auto count = n.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        fgp_accumulate_inplace_kernel<FreshT, SumT, SquareT, uint32_t>(
            sum_info, square_info, count_info, fresh_info, weight);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(n)) {
        auto count = n.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        fgp_accumulate_inplace_kernel<FreshT, SumT, SquareT, uint64_t>(
            sum_info, square_info, count_info, fresh_info, weight);
        return;
    }
    if (py::isinstance<py::array_t<double>>(n)) {
        auto count = n.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        fgp_accumulate_inplace_kernel<FreshT, SumT, SquareT, double>(
            sum_info, square_info, count_info, fresh_info, weight);
        return;
    }
    throw std::invalid_argument("fgp_accumulate: unsupported n dtype");
}

template <typename FreshT, typename SumT>
void dispatch_accumulate_square_dtype(
    py::array_t<SumT, py::array::c_style | py::array::forcecast> sum_mu,
    py::array square_sum,
    py::array n,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& fresh,
    const uint64_t weight) {
    if (py::isinstance<py::array_t<uint32_t>>(square_sum)) {
        dispatch_accumulate_count_dtype<FreshT, SumT, uint32_t>(
            sum_mu,
            square_sum.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>(),
            n,
            fresh,
            weight);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(square_sum)) {
        dispatch_accumulate_count_dtype<FreshT, SumT, uint64_t>(
            sum_mu,
            square_sum.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>(),
            n,
            fresh,
            weight);
        return;
    }
    if (py::isinstance<py::array_t<double>>(square_sum)) {
        dispatch_accumulate_count_dtype<FreshT, SumT, double>(
            sum_mu,
            square_sum.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(),
            n,
            fresh,
            weight);
        return;
    }
    throw std::invalid_argument("fgp_accumulate: unsupported square_sum dtype");
}

template <typename FreshT>
void dispatch_accumulate_sum_dtype(
    py::array sum_mu,
    py::array square_sum,
    py::array n,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& fresh,
    const uint64_t weight) {
    if (py::isinstance<py::array_t<uint16_t>>(sum_mu)) {
        dispatch_accumulate_square_dtype<FreshT, uint16_t>(
            sum_mu.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(),
            square_sum,
            n,
            fresh,
            weight);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(sum_mu)) {
        dispatch_accumulate_square_dtype<FreshT, uint32_t>(
            sum_mu.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>(),
            square_sum,
            n,
            fresh,
            weight);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(sum_mu)) {
        dispatch_accumulate_square_dtype<FreshT, uint64_t>(
            sum_mu.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>(),
            square_sum,
            n,
            fresh,
            weight);
        return;
    }
    if (py::isinstance<py::array_t<double>>(sum_mu)) {
        dispatch_accumulate_square_dtype<FreshT, double>(
            sum_mu.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(),
            square_sum,
            n,
            fresh,
            weight);
        return;
    }
    throw std::invalid_argument("fgp_accumulate: unsupported sum_mu dtype");
}

template <typename SumT, typename SquareT>
void dispatch_add_count_dtype(
    py::array_t<SumT, py::array::c_style | py::array::forcecast> sum_mu,
    py::array_t<SquareT, py::array::c_style | py::array::forcecast> square_sum,
    py::array n,
    const py::array& other_sum_mu,
    const py::array& other_square_sum,
    const py::array& other_n) {
    if (py::isinstance<py::array_t<uint16_t>>(n)) {
        auto count = n.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>();
        auto other_count =
            other_n.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>();
        auto other_sum =
            other_sum_mu.cast<py::array_t<SumT, py::array::c_style | py::array::forcecast>>();
        auto other_square = other_square_sum.cast<
            py::array_t<SquareT, py::array::c_style | py::array::forcecast>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto other_sum_info = other_sum.request();
        auto other_square_info = other_square.request();
        auto other_count_info = other_count.request();
        fgp_add_inplace_kernel<SumT, SquareT, uint16_t>(
            sum_info, square_info, count_info, other_sum_info, other_square_info, other_count_info);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(n)) {
        auto count = n.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>();
        auto other_count =
            other_n.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>();
        auto other_sum =
            other_sum_mu.cast<py::array_t<SumT, py::array::c_style | py::array::forcecast>>();
        auto other_square = other_square_sum.cast<
            py::array_t<SquareT, py::array::c_style | py::array::forcecast>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto other_sum_info = other_sum.request();
        auto other_square_info = other_square.request();
        auto other_count_info = other_count.request();
        fgp_add_inplace_kernel<SumT, SquareT, uint32_t>(
            sum_info, square_info, count_info, other_sum_info, other_square_info, other_count_info);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(n)) {
        auto count = n.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>();
        auto other_count =
            other_n.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>();
        auto other_sum =
            other_sum_mu.cast<py::array_t<SumT, py::array::c_style | py::array::forcecast>>();
        auto other_square = other_square_sum.cast<
            py::array_t<SquareT, py::array::c_style | py::array::forcecast>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto other_sum_info = other_sum.request();
        auto other_square_info = other_square.request();
        auto other_count_info = other_count.request();
        fgp_add_inplace_kernel<SumT, SquareT, uint64_t>(
            sum_info, square_info, count_info, other_sum_info, other_square_info, other_count_info);
        return;
    }
    if (py::isinstance<py::array_t<double>>(n)) {
        auto count = n.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        auto other_count =
            other_n.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        auto other_sum =
            other_sum_mu.cast<py::array_t<SumT, py::array::c_style | py::array::forcecast>>();
        auto other_square = other_square_sum.cast<
            py::array_t<SquareT, py::array::c_style | py::array::forcecast>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto other_sum_info = other_sum.request();
        auto other_square_info = other_square.request();
        auto other_count_info = other_count.request();
        fgp_add_inplace_kernel<SumT, SquareT, double>(
            sum_info, square_info, count_info, other_sum_info, other_square_info, other_count_info);
        return;
    }
    throw std::invalid_argument("fgp_add: unsupported n dtype");
}

template <typename SumT>
void dispatch_add_square_dtype(
    py::array_t<SumT, py::array::c_style | py::array::forcecast> sum_mu,
    py::array square_sum,
    py::array n,
    const py::array& other_sum_mu,
    const py::array& other_square_sum,
    const py::array& other_n) {
    if (py::isinstance<py::array_t<uint32_t>>(square_sum)) {
        dispatch_add_count_dtype<SumT, uint32_t>(
            sum_mu,
            square_sum.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>(),
            n,
            other_sum_mu,
            other_square_sum,
            other_n);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(square_sum)) {
        dispatch_add_count_dtype<SumT, uint64_t>(
            sum_mu,
            square_sum.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>(),
            n,
            other_sum_mu,
            other_square_sum,
            other_n);
        return;
    }
    if (py::isinstance<py::array_t<double>>(square_sum)) {
        dispatch_add_count_dtype<SumT, double>(
            sum_mu,
            square_sum.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(),
            n,
            other_sum_mu,
            other_square_sum,
            other_n);
        return;
    }
    throw std::invalid_argument("fgp_add: unsupported square_sum dtype");
}

void dispatch_add_sum_dtype(py::array sum_mu,
                            py::array square_sum,
                            py::array n,
                            const py::array& other_sum_mu,
                            const py::array& other_square_sum,
                            const py::array& other_n) {
    if (py::isinstance<py::array_t<uint16_t>>(sum_mu)) {
        dispatch_add_square_dtype<uint16_t>(
            sum_mu.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(),
            square_sum,
            n,
            other_sum_mu,
            other_square_sum,
            other_n);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(sum_mu)) {
        dispatch_add_square_dtype<uint32_t>(
            sum_mu.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>(),
            square_sum,
            n,
            other_sum_mu,
            other_square_sum,
            other_n);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(sum_mu)) {
        dispatch_add_square_dtype<uint64_t>(
            sum_mu.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>(),
            square_sum,
            n,
            other_sum_mu,
            other_square_sum,
            other_n);
        return;
    }
    if (py::isinstance<py::array_t<double>>(sum_mu)) {
        dispatch_add_square_dtype<double>(
            sum_mu.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(),
            square_sum,
            n,
            other_sum_mu,
            other_square_sum,
            other_n);
        return;
    }
    throw std::invalid_argument("fgp_add: unsupported sum_mu dtype");
}

template <typename FreshT, typename SumT, typename SquareT>
void dispatch_masked_mean_count_dtype(
    py::array_t<SumT, py::array::c_style | py::array::forcecast> sum_mu,
    py::array_t<SquareT, py::array::c_style | py::array::forcecast> square_sum,
    py::array n,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& fresh,
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& mask) {
    if (py::isinstance<py::array_t<uint16_t>>(n)) {
        auto count = n.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        auto mask_info = mask.request();
        fgp_masked_mean_inplace_kernel<FreshT, SumT, SquareT, uint16_t>(
            sum_info, square_info, count_info, fresh_info, mask_info);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(n)) {
        auto count = n.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        auto mask_info = mask.request();
        fgp_masked_mean_inplace_kernel<FreshT, SumT, SquareT, uint32_t>(
            sum_info, square_info, count_info, fresh_info, mask_info);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(n)) {
        auto count = n.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        auto mask_info = mask.request();
        fgp_masked_mean_inplace_kernel<FreshT, SumT, SquareT, uint64_t>(
            sum_info, square_info, count_info, fresh_info, mask_info);
        return;
    }
    if (py::isinstance<py::array_t<double>>(n)) {
        auto count = n.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        auto mask_info = mask.request();
        fgp_masked_mean_inplace_kernel<FreshT, SumT, SquareT, double>(
            sum_info, square_info, count_info, fresh_info, mask_info);
        return;
    }
    throw std::invalid_argument("fgp_masked_mean_merge: unsupported n dtype");
}

template <typename FreshT, typename SumT>
void dispatch_masked_mean_square_dtype(
    py::array_t<SumT, py::array::c_style | py::array::forcecast> sum_mu,
    py::array square_sum,
    py::array n,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& fresh,
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& mask) {
    if (py::isinstance<py::array_t<uint32_t>>(square_sum)) {
        dispatch_masked_mean_count_dtype<FreshT, SumT, uint32_t>(
            sum_mu,
            square_sum.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>(),
            n,
            fresh,
            mask);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(square_sum)) {
        dispatch_masked_mean_count_dtype<FreshT, SumT, uint64_t>(
            sum_mu,
            square_sum.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>(),
            n,
            fresh,
            mask);
        return;
    }
    if (py::isinstance<py::array_t<double>>(square_sum)) {
        dispatch_masked_mean_count_dtype<FreshT, SumT, double>(
            sum_mu,
            square_sum.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(),
            n,
            fresh,
            mask);
        return;
    }
    throw std::invalid_argument("fgp_masked_mean_merge: unsupported square_sum dtype");
}

template <typename FreshT>
void dispatch_masked_mean_sum_dtype(
    py::array sum_mu,
    py::array square_sum,
    py::array n,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& fresh,
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& mask) {
    if (py::isinstance<py::array_t<uint16_t>>(sum_mu)) {
        dispatch_masked_mean_square_dtype<FreshT, uint16_t>(
            sum_mu.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(),
            square_sum,
            n,
            fresh,
            mask);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(sum_mu)) {
        dispatch_masked_mean_square_dtype<FreshT, uint32_t>(
            sum_mu.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>(),
            square_sum,
            n,
            fresh,
            mask);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(sum_mu)) {
        dispatch_masked_mean_square_dtype<FreshT, uint64_t>(
            sum_mu.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>(),
            square_sum,
            n,
            fresh,
            mask);
        return;
    }
    if (py::isinstance<py::array_t<double>>(sum_mu)) {
        dispatch_masked_mean_square_dtype<FreshT, double>(
            sum_mu.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(),
            square_sum,
            n,
            fresh,
            mask);
        return;
    }
    throw std::invalid_argument("fgp_masked_mean_merge: unsupported sum_mu dtype");
}

template <typename FreshT, typename SumT, typename SquareT>
void dispatch_sigma_clip_count_dtype(
    py::array_t<SumT, py::array::c_style | py::array::forcecast> sum_mu,
    py::array_t<SquareT, py::array::c_style | py::array::forcecast> square_sum,
    py::array n,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& fresh,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& rej_high,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& rej_low) {
    if (py::isinstance<py::array_t<uint16_t>>(n)) {
        auto count = n.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        auto rej_high_info = rej_high.request();
        auto rej_low_info = rej_low.request();
        sigma_clip_fused_inplace_kernel<FreshT, SumT, SquareT, uint16_t>(
            sum_info, square_info, count_info, fresh_info, rej_high_info, rej_low_info);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(n)) {
        auto count = n.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        auto rej_high_info = rej_high.request();
        auto rej_low_info = rej_low.request();
        sigma_clip_fused_inplace_kernel<FreshT, SumT, SquareT, uint32_t>(
            sum_info, square_info, count_info, fresh_info, rej_high_info, rej_low_info);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(n)) {
        auto count = n.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        auto rej_high_info = rej_high.request();
        auto rej_low_info = rej_low.request();
        sigma_clip_fused_inplace_kernel<FreshT, SumT, SquareT, uint64_t>(
            sum_info, square_info, count_info, fresh_info, rej_high_info, rej_low_info);
        return;
    }
    if (py::isinstance<py::array_t<double>>(n)) {
        auto count = n.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        auto rej_high_info = rej_high.request();
        auto rej_low_info = rej_low.request();
        sigma_clip_fused_inplace_kernel<FreshT, SumT, SquareT, double>(
            sum_info, square_info, count_info, fresh_info, rej_high_info, rej_low_info);
        return;
    }
    throw std::invalid_argument("sigma_clip_fused_merge: unsupported n dtype");
}

template <typename FreshT, typename SumT>
void dispatch_sigma_clip_square_dtype(
    py::array_t<SumT, py::array::c_style | py::array::forcecast> sum_mu,
    py::array square_sum,
    py::array n,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& fresh,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& rej_high,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& rej_low) {
    if (py::isinstance<py::array_t<uint32_t>>(square_sum)) {
        dispatch_sigma_clip_count_dtype<FreshT, SumT, uint32_t>(
            sum_mu,
            square_sum.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>(),
            n,
            fresh,
            rej_high,
            rej_low);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(square_sum)) {
        dispatch_sigma_clip_count_dtype<FreshT, SumT, uint64_t>(
            sum_mu,
            square_sum.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>(),
            n,
            fresh,
            rej_high,
            rej_low);
        return;
    }
    if (py::isinstance<py::array_t<double>>(square_sum)) {
        dispatch_sigma_clip_count_dtype<FreshT, SumT, double>(
            sum_mu,
            square_sum.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(),
            n,
            fresh,
            rej_high,
            rej_low);
        return;
    }
    throw std::invalid_argument("sigma_clip_fused_merge: unsupported square_sum dtype");
}

template <typename FreshT>
void dispatch_sigma_clip_sum_dtype(
    py::array sum_mu,
    py::array square_sum,
    py::array n,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& fresh,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& rej_high,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& rej_low) {
    if (py::isinstance<py::array_t<uint16_t>>(sum_mu)) {
        dispatch_sigma_clip_square_dtype<FreshT, uint16_t>(
            sum_mu.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(),
            square_sum,
            n,
            fresh,
            rej_high,
            rej_low);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(sum_mu)) {
        dispatch_sigma_clip_square_dtype<FreshT, uint32_t>(
            sum_mu.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>(),
            square_sum,
            n,
            fresh,
            rej_high,
            rej_low);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(sum_mu)) {
        dispatch_sigma_clip_square_dtype<FreshT, uint64_t>(
            sum_mu.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>(),
            square_sum,
            n,
            fresh,
            rej_high,
            rej_low);
        return;
    }
    if (py::isinstance<py::array_t<double>>(sum_mu)) {
        dispatch_sigma_clip_square_dtype<FreshT, double>(
            sum_mu.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(),
            square_sum,
            n,
            fresh,
            rej_high,
            rej_low);
        return;
    }
    throw std::invalid_argument("sigma_clip_fused_merge: unsupported sum_mu dtype");
}

template <typename FreshT, typename SumT, typename SquareT>
void dispatch_sigma_clip_masked_count_dtype(
    py::array_t<SumT, py::array::c_style | py::array::forcecast> sum_mu,
    py::array_t<SquareT, py::array::c_style | py::array::forcecast> square_sum,
    py::array n,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& fresh,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& rej_high,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& rej_low,
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& mask) {
    if (py::isinstance<py::array_t<uint16_t>>(n)) {
        auto count = n.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        auto rej_high_info = rej_high.request();
        auto rej_low_info = rej_low.request();
        auto mask_info = mask.request();
        sigma_clip_fused_masked_inplace_kernel<FreshT, SumT, SquareT, uint16_t>(
            sum_info, square_info, count_info, fresh_info, rej_high_info, rej_low_info, mask_info);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(n)) {
        auto count = n.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        auto rej_high_info = rej_high.request();
        auto rej_low_info = rej_low.request();
        auto mask_info = mask.request();
        sigma_clip_fused_masked_inplace_kernel<FreshT, SumT, SquareT, uint32_t>(
            sum_info, square_info, count_info, fresh_info, rej_high_info, rej_low_info, mask_info);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(n)) {
        auto count = n.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        auto rej_high_info = rej_high.request();
        auto rej_low_info = rej_low.request();
        auto mask_info = mask.request();
        sigma_clip_fused_masked_inplace_kernel<FreshT, SumT, SquareT, uint64_t>(
            sum_info, square_info, count_info, fresh_info, rej_high_info, rej_low_info, mask_info);
        return;
    }
    if (py::isinstance<py::array_t<double>>(n)) {
        auto count = n.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        auto sum_info = sum_mu.request();
        auto square_info = square_sum.request();
        auto count_info = count.request();
        auto fresh_info = fresh.request();
        auto rej_high_info = rej_high.request();
        auto rej_low_info = rej_low.request();
        auto mask_info = mask.request();
        sigma_clip_fused_masked_inplace_kernel<FreshT, SumT, SquareT, double>(
            sum_info, square_info, count_info, fresh_info, rej_high_info, rej_low_info, mask_info);
        return;
    }
    throw std::invalid_argument("sigma_clip_fused_masked_merge: unsupported n dtype");
}

template <typename FreshT, typename SumT>
void dispatch_sigma_clip_masked_square_dtype(
    py::array_t<SumT, py::array::c_style | py::array::forcecast> sum_mu,
    py::array square_sum,
    py::array n,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& fresh,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& rej_high,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& rej_low,
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& mask) {
    if (py::isinstance<py::array_t<uint32_t>>(square_sum)) {
        dispatch_sigma_clip_masked_count_dtype<FreshT, SumT, uint32_t>(
            sum_mu,
            square_sum.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>(),
            n,
            fresh,
            rej_high,
            rej_low,
            mask);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(square_sum)) {
        dispatch_sigma_clip_masked_count_dtype<FreshT, SumT, uint64_t>(
            sum_mu,
            square_sum.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>(),
            n,
            fresh,
            rej_high,
            rej_low,
            mask);
        return;
    }
    if (py::isinstance<py::array_t<double>>(square_sum)) {
        dispatch_sigma_clip_masked_count_dtype<FreshT, SumT, double>(
            sum_mu,
            square_sum.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(),
            n,
            fresh,
            rej_high,
            rej_low,
            mask);
        return;
    }
    throw std::invalid_argument("sigma_clip_fused_masked_merge: unsupported square_sum dtype");
}

template <typename FreshT>
void dispatch_sigma_clip_masked_sum_dtype(
    py::array sum_mu,
    py::array square_sum,
    py::array n,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& fresh,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& rej_high,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& rej_low,
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& mask) {
    if (py::isinstance<py::array_t<uint16_t>>(sum_mu)) {
        dispatch_sigma_clip_masked_square_dtype<FreshT, uint16_t>(
            sum_mu.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(),
            square_sum,
            n,
            fresh,
            rej_high,
            rej_low,
            mask);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(sum_mu)) {
        dispatch_sigma_clip_masked_square_dtype<FreshT, uint32_t>(
            sum_mu.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>(),
            square_sum,
            n,
            fresh,
            rej_high,
            rej_low,
            mask);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(sum_mu)) {
        dispatch_sigma_clip_masked_square_dtype<FreshT, uint64_t>(
            sum_mu.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>(),
            square_sum,
            n,
            fresh,
            rej_high,
            rej_low,
            mask);
        return;
    }
    if (py::isinstance<py::array_t<double>>(sum_mu)) {
        dispatch_sigma_clip_masked_square_dtype<FreshT, double>(
            sum_mu.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(),
            square_sum,
            n,
            fresh,
            rej_high,
            rej_low,
            mask);
        return;
    }
    throw std::invalid_argument("sigma_clip_fused_masked_merge: unsupported sum_mu dtype");
}

template <typename FreshT>
void dispatch_huber_weighted_accumulate(
    py::array weighted_sum,
    py::array weight_total,
    const py::array_t<FreshT, py::array::c_style | py::array::forcecast>& fresh,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& ref_mean,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& ref_std,
    const double huber_c,
    const double frame_weight) {
    auto weighted_sum_t =
        weighted_sum.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    auto weight_total_t =
        weight_total.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    auto weighted_sum_info = weighted_sum_t.request();
    auto weight_total_info = weight_total_t.request();
    auto fresh_info = fresh.request();
    auto ref_mean_info = ref_mean.request();
    auto ref_std_info = ref_std.request();
    huber_weighted_accumulate_inplace_kernel<FreshT>(
        weighted_sum_info,
        weight_total_info,
        fresh_info,
        ref_mean_info,
        ref_std_info,
        huber_c,
        frame_weight);
}

void validate_accumulate_shapes(const py::array& sum_mu,
                                const py::array& square_sum,
                                const py::array& n,
                                const py::array& fresh) {
    validate_accumulator_shapes(sum_mu, square_sum, n, fresh, "fgp_accumulate");
}

void validate_add_shapes(const py::array& sum_mu,
                         const py::array& square_sum,
                         const py::array& n,
                         const py::array& other_sum_mu,
                         const py::array& other_square_sum,
                         const py::array& other_n) {
    validate_accumulator_shapes(sum_mu, square_sum, n, other_sum_mu, "fgp_add");
    if (other_square_sum.ndim() != other_sum_mu.ndim() ||
        other_n.ndim() != other_sum_mu.ndim()) {
        throw std::invalid_argument("fgp_add: ndim mismatch");
    }
    for (ssize_t i = 0; i < other_sum_mu.ndim(); ++i) {
        if (other_square_sum.shape(i) != other_sum_mu.shape(i) ||
            other_n.shape(i) != other_sum_mu.shape(i)) {
            throw std::invalid_argument("fgp_add: shape mismatch");
        }
    }
}

void validate_masked_shapes(const py::array& sum_mu,
                            const py::array& square_sum,
                            const py::array& n,
                            const py::array& fresh,
                            const py::array& mask,
                            const char* op_name) {
    validate_accumulator_shapes(sum_mu, square_sum, n, fresh, op_name);
    if (mask.ndim() == fresh.ndim()) {
        for (ssize_t i = 0; i < fresh.ndim(); ++i) {
            if (mask.shape(i) != fresh.shape(i)) {
                throw std::invalid_argument(std::string(op_name) + ": mask shape mismatch");
            }
        }
        return;
    }
    if (mask.ndim() + 1 == fresh.ndim()) {
        for (ssize_t i = 0; i < mask.ndim(); ++i) {
            if (mask.shape(i) != fresh.shape(i)) {
                throw std::invalid_argument(std::string(op_name) + ": mask shape mismatch");
            }
        }
        return;
    }
    throw std::invalid_argument(std::string(op_name) + ": mask ndim mismatch");
}

void validate_sigma_shapes(const py::array& sum_mu,
                           const py::array& square_sum,
                           const py::array& n,
                           const py::array& fresh,
                           const py::array& rej_high,
                           const py::array& rej_low,
                           const char* op_name) {
    validate_accumulator_shapes(sum_mu, square_sum, n, fresh, op_name);
    if (rej_high.ndim() != fresh.ndim() || rej_low.ndim() != fresh.ndim()) {
        throw std::invalid_argument(std::string(op_name) + ": rejection image ndim mismatch");
    }
    for (ssize_t i = 0; i < fresh.ndim(); ++i) {
        if (rej_high.shape(i) != fresh.shape(i) ||
            rej_low.shape(i) != fresh.shape(i)) {
            throw std::invalid_argument(std::string(op_name) + ": rejection image shape mismatch");
        }
    }
    if (py::str(rej_high.dtype()).cast<std::string>() !=
            py::str(fresh.dtype()).cast<std::string>() ||
        py::str(rej_low.dtype()).cast<std::string>() !=
            py::str(fresh.dtype()).cast<std::string>()) {
        throw std::invalid_argument(std::string(op_name) + ": rejection image dtype mismatch");
    }
}

void validate_huber_shapes(const py::array& weighted_sum,
                           const py::array& weight_total,
                           const py::array& fresh,
                           const py::array& ref_mean,
                           const py::array& ref_std) {
    if (py::str(weighted_sum.dtype()).cast<std::string>() != "float64" ||
        py::str(weight_total.dtype()).cast<std::string>() != "float64") {
        throw std::invalid_argument(
            "huber_weighted_accumulate: accumulators must be float64");
    }
    if (weighted_sum.ndim() != fresh.ndim() || weight_total.ndim() != fresh.ndim() ||
        ref_mean.ndim() != fresh.ndim() || ref_std.ndim() != fresh.ndim()) {
        throw std::invalid_argument("huber_weighted_accumulate: ndim mismatch");
    }
    for (ssize_t i = 0; i < fresh.ndim(); ++i) {
        if (weighted_sum.shape(i) != fresh.shape(i) ||
            weight_total.shape(i) != fresh.shape(i) ||
            ref_mean.shape(i) != fresh.shape(i) ||
            ref_std.shape(i) != fresh.shape(i)) {
            throw std::invalid_argument("huber_weighted_accumulate: shape mismatch");
        }
    }
}

void fgp_accumulate_dispatch(py::array sum_mu,
                             py::array square_sum,
                             py::array n,
                             const py::array& fresh,
                             const py::object& weight_obj) {
    validate_accumulate_shapes(sum_mu, square_sum, n, fresh);
    uint64_t weight = 1;
    if (!weight_obj.is_none()) {
        weight = py::cast<uint64_t>(weight_obj);
    }

    if (py::isinstance<py::array_t<uint8_t>>(fresh)) {
        dispatch_accumulate_sum_dtype<uint8_t>(
            sum_mu, square_sum, n,
            fresh.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(),
            weight);
        return;
    }
    if (py::isinstance<py::array_t<uint16_t>>(fresh)) {
        dispatch_accumulate_sum_dtype<uint16_t>(
            sum_mu, square_sum, n,
            fresh.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(),
            weight);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(fresh)) {
        dispatch_accumulate_sum_dtype<uint32_t>(
            sum_mu, square_sum, n,
            fresh.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>(),
            weight);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(fresh)) {
        dispatch_accumulate_sum_dtype<uint64_t>(
            sum_mu, square_sum, n,
            fresh.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>(),
            weight);
        return;
    }
    if (py::isinstance<py::array_t<float>>(fresh)) {
        dispatch_accumulate_sum_dtype<float>(
            sum_mu, square_sum, n,
            fresh.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(),
            weight);
        return;
    }
    if (py::isinstance<py::array_t<double>>(fresh)) {
        dispatch_accumulate_sum_dtype<double>(
            sum_mu, square_sum, n,
            fresh.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(),
            weight);
        return;
    }
    throw std::invalid_argument("fgp_accumulate: unsupported fresh dtype");
}

void fgp_add_dispatch(py::array sum_mu,
                      py::array square_sum,
                      py::array n,
                      const py::array& other_sum_mu,
                      const py::array& other_square_sum,
                      const py::array& other_n) {
    validate_add_shapes(sum_mu, square_sum, n, other_sum_mu, other_square_sum,
                        other_n);
    dispatch_add_sum_dtype(
        sum_mu, square_sum, n, other_sum_mu, other_square_sum, other_n);
}

void fgp_masked_mean_dispatch(py::array sum_mu,
                              py::array square_sum,
                              py::array n,
                              const py::array& fresh,
                              const py::array& mask) {
    validate_masked_shapes(sum_mu, square_sum, n, fresh, mask,
                           "fgp_masked_mean_merge");
    auto mask_t =
        mask.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();

    if (py::isinstance<py::array_t<uint8_t>>(fresh)) {
        dispatch_masked_mean_sum_dtype<uint8_t>(
            sum_mu, square_sum, n,
            fresh.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(),
            mask_t);
        return;
    }
    if (py::isinstance<py::array_t<uint16_t>>(fresh)) {
        dispatch_masked_mean_sum_dtype<uint16_t>(
            sum_mu, square_sum, n,
            fresh.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(),
            mask_t);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(fresh)) {
        dispatch_masked_mean_sum_dtype<uint32_t>(
            sum_mu, square_sum, n,
            fresh.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>(),
            mask_t);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(fresh)) {
        dispatch_masked_mean_sum_dtype<uint64_t>(
            sum_mu, square_sum, n,
            fresh.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>(),
            mask_t);
        return;
    }
    if (py::isinstance<py::array_t<float>>(fresh)) {
        dispatch_masked_mean_sum_dtype<float>(
            sum_mu, square_sum, n,
            fresh.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(),
            mask_t);
        return;
    }
    if (py::isinstance<py::array_t<double>>(fresh)) {
        dispatch_masked_mean_sum_dtype<double>(
            sum_mu, square_sum, n,
            fresh.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(),
            mask_t);
        return;
    }
    throw std::invalid_argument("fgp_masked_mean_merge: unsupported fresh dtype");
}

void sigma_clip_fused_dispatch(py::array sum_mu,
                               py::array square_sum,
                               py::array n,
                               const py::array& fresh,
                               const py::array& rej_high,
                               const py::array& rej_low) {
    validate_sigma_shapes(sum_mu, square_sum, n, fresh, rej_high, rej_low,
                          "sigma_clip_fused_merge");

    if (py::isinstance<py::array_t<uint8_t>>(fresh)) {
        auto fresh_t =
            fresh.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
        dispatch_sigma_clip_sum_dtype<uint8_t>(
            sum_mu,
            square_sum,
            n,
            fresh_t,
            rej_high.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(),
            rej_low.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>());
        return;
    }
    if (py::isinstance<py::array_t<uint16_t>>(fresh)) {
        auto fresh_t =
            fresh.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>();
        dispatch_sigma_clip_sum_dtype<uint16_t>(
            sum_mu,
            square_sum,
            n,
            fresh_t,
            rej_high.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(),
            rej_low.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>());
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(fresh)) {
        auto fresh_t =
            fresh.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>();
        dispatch_sigma_clip_sum_dtype<uint32_t>(
            sum_mu,
            square_sum,
            n,
            fresh_t,
            rej_high.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>(),
            rej_low.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>());
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(fresh)) {
        auto fresh_t =
            fresh.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>();
        dispatch_sigma_clip_sum_dtype<uint64_t>(
            sum_mu,
            square_sum,
            n,
            fresh_t,
            rej_high.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>(),
            rej_low.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>());
        return;
    }
    if (py::isinstance<py::array_t<float>>(fresh)) {
        auto fresh_t =
            fresh.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        dispatch_sigma_clip_sum_dtype<float>(
            sum_mu,
            square_sum,
            n,
            fresh_t,
            rej_high.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(),
            rej_low.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>());
        return;
    }
    if (py::isinstance<py::array_t<double>>(fresh)) {
        auto fresh_t =
            fresh.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        dispatch_sigma_clip_sum_dtype<double>(
            sum_mu,
            square_sum,
            n,
            fresh_t,
            rej_high.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(),
            rej_low.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>());
        return;
    }
    throw std::invalid_argument("sigma_clip_fused_merge: unsupported fresh dtype");
}

void sigma_clip_fused_masked_dispatch(py::array sum_mu,
                                      py::array square_sum,
                                      py::array n,
                                      const py::array& fresh,
                                      const py::array& rej_high,
                                      const py::array& rej_low,
                                      const py::array& mask) {
    validate_sigma_shapes(sum_mu, square_sum, n, fresh, rej_high, rej_low,
                          "sigma_clip_fused_masked_merge");
    validate_masked_shapes(sum_mu, square_sum, n, fresh, mask,
                           "sigma_clip_fused_masked_merge");
    auto mask_t =
        mask.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();

    if (py::isinstance<py::array_t<uint8_t>>(fresh)) {
        auto fresh_t =
            fresh.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
        dispatch_sigma_clip_masked_sum_dtype<uint8_t>(
            sum_mu,
            square_sum,
            n,
            fresh_t,
            rej_high.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(),
            rej_low.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(),
            mask_t);
        return;
    }
    if (py::isinstance<py::array_t<uint16_t>>(fresh)) {
        auto fresh_t =
            fresh.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>();
        dispatch_sigma_clip_masked_sum_dtype<uint16_t>(
            sum_mu,
            square_sum,
            n,
            fresh_t,
            rej_high.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(),
            rej_low.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(),
            mask_t);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(fresh)) {
        auto fresh_t =
            fresh.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>();
        dispatch_sigma_clip_masked_sum_dtype<uint32_t>(
            sum_mu,
            square_sum,
            n,
            fresh_t,
            rej_high.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>(),
            rej_low.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>(),
            mask_t);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(fresh)) {
        auto fresh_t =
            fresh.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>();
        dispatch_sigma_clip_masked_sum_dtype<uint64_t>(
            sum_mu,
            square_sum,
            n,
            fresh_t,
            rej_high.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>(),
            rej_low.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>(),
            mask_t);
        return;
    }
    if (py::isinstance<py::array_t<float>>(fresh)) {
        auto fresh_t =
            fresh.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        dispatch_sigma_clip_masked_sum_dtype<float>(
            sum_mu,
            square_sum,
            n,
            fresh_t,
            rej_high.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(),
            rej_low.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(),
            mask_t);
        return;
    }
    if (py::isinstance<py::array_t<double>>(fresh)) {
        auto fresh_t =
            fresh.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        dispatch_sigma_clip_masked_sum_dtype<double>(
            sum_mu,
            square_sum,
            n,
            fresh_t,
            rej_high.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(),
            rej_low.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(),
            mask_t);
        return;
    }
    throw std::invalid_argument("sigma_clip_fused_masked_merge: unsupported fresh dtype");
}

void huber_weighted_accumulate_dispatch(py::array weighted_sum,
                                        py::array weight_total,
                                        const py::array& fresh,
                                        const py::array& ref_mean,
                                        const py::array& ref_std,
                                        double huber_c,
                                        const py::object& weight_obj) {
    validate_huber_shapes(weighted_sum, weight_total, fresh, ref_mean, ref_std);
    double frame_weight = 1.0;
    if (!weight_obj.is_none()) {
        frame_weight = py::cast<double>(weight_obj);
    }
    auto ref_mean_t =
        ref_mean.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto ref_std_t =
        ref_std.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();

    if (py::isinstance<py::array_t<uint8_t>>(fresh)) {
        dispatch_huber_weighted_accumulate<uint8_t>(
            weighted_sum,
            weight_total,
            fresh.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(),
            ref_mean_t,
            ref_std_t,
            huber_c,
            frame_weight);
        return;
    }
    if (py::isinstance<py::array_t<uint16_t>>(fresh)) {
        dispatch_huber_weighted_accumulate<uint16_t>(
            weighted_sum,
            weight_total,
            fresh.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>(),
            ref_mean_t,
            ref_std_t,
            huber_c,
            frame_weight);
        return;
    }
    if (py::isinstance<py::array_t<uint32_t>>(fresh)) {
        dispatch_huber_weighted_accumulate<uint32_t>(
            weighted_sum,
            weight_total,
            fresh.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>(),
            ref_mean_t,
            ref_std_t,
            huber_c,
            frame_weight);
        return;
    }
    if (py::isinstance<py::array_t<uint64_t>>(fresh)) {
        dispatch_huber_weighted_accumulate<uint64_t>(
            weighted_sum,
            weight_total,
            fresh.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>(),
            ref_mean_t,
            ref_std_t,
            huber_c,
            frame_weight);
        return;
    }
    if (py::isinstance<py::array_t<float>>(fresh)) {
        dispatch_huber_weighted_accumulate<float>(
            weighted_sum,
            weight_total,
            fresh.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(),
            ref_mean_t,
            ref_std_t,
            huber_c,
            frame_weight);
        return;
    }
    if (py::isinstance<py::array_t<double>>(fresh)) {
        dispatch_huber_weighted_accumulate<double>(
            weighted_sum,
            weight_total,
            fresh.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(),
            ref_mean_t,
            ref_std_t,
            huber_c,
            frame_weight);
        return;
    }
    throw std::invalid_argument("huber_weighted_accumulate: unsupported fresh dtype");
}

}  // namespace

void bind_fgp_ops(py::module_& m) {
    m.def(
        "fgp_accumulate",
        &fgp_accumulate_dispatch,
        py::arg("sum_mu"),
        py::arg("square_sum"),
        py::arg("n"),
        py::arg("fresh"),
        py::arg("weight") = py::none(),
        "Update FastGaussianParam buffers in-place with one more frame.");
    m.def(
        "fgp_add",
        &fgp_add_dispatch,
        py::arg("sum_mu"),
        py::arg("square_sum"),
        py::arg("n"),
        py::arg("other_sum_mu"),
        py::arg("other_square_sum"),
        py::arg("other_n"),
        "Update FastGaussianParam buffers in-place with another FastGaussianParam.");
    m.def(
        "huber_weighted_accumulate",
        &huber_weighted_accumulate_dispatch,
        py::arg("weighted_sum"),
        py::arg("weight_total"),
        py::arg("fresh"),
        py::arg("ref_mean"),
        py::arg("ref_std"),
        py::arg("huber_c"),
        py::arg("weight") = py::none(),
        "Update HuberMeanParam buffers in-place with one more frame.");
    m.def(
        "fgp_masked_mean_merge",
        &fgp_masked_mean_dispatch,
        py::arg("sum_mu"),
        py::arg("square_sum"),
        py::arg("n"),
        py::arg("fresh"),
        py::arg("mask"),
        "Update FastGaussianParam buffers in-place for spatial-mask mean merge.");
    m.def(
        "sigma_clip_fused_merge",
        &sigma_clip_fused_dispatch,
        py::arg("sum_mu"),
        py::arg("square_sum"),
        py::arg("n"),
        py::arg("fresh"),
        py::arg("rej_high"),
        py::arg("rej_low"),
        "Update rejected FastGaussianParam buffers in-place for sigma clip.");
    m.def(
        "sigma_clip_fused_masked_merge",
        &sigma_clip_fused_masked_dispatch,
        py::arg("sum_mu"),
        py::arg("square_sum"),
        py::arg("n"),
        py::arg("fresh"),
        py::arg("rej_high"),
        py::arg("rej_low"),
        py::arg("mask"),
        "Update rejected FastGaussianParam buffers in-place for masked sigma clip.");
}
