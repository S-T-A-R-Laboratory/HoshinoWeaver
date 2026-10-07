#pragma once

#include "common/compat.h"

#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace hnw {

class StarDetectCapacityError : public std::runtime_error {
public:
    explicit StarDetectCapacityError(const std::string& message) : std::runtime_error(message) {}
};

} // namespace hnw

// Host image read by the detector: a float64 gray plane, or a uint8/uint16
// image with 1 or 3 (BGR) channels that the device converts to gray exactly as
// to_gray_f64 does through OpenCV/IPP.
struct StarDetectSource {
    const void* data = nullptr;
    int channels = 1;
    int sample_bytes = sizeof(double);

    bool is_gray() const { return sample_bytes == sizeof(double); }
};

// Returns false, leaving the outputs untouched, when a converted source has a
// constant gray (the float64 gray path checks that on the host).
bool launch_star_detect_fused_pixel_components(
    const StarDetectSource& source, const uint8_t* external_mask_host,
    const double* gaussian_kernel_host, std::vector<double>* positions_xy_host,
    std::vector<double>* intensities_host, uint8_t* binary_mask_host, int height, int width,
    int small_height, int small_width, int level, int gaussian_ksize);

// Device gray of a uint8/uint16 source, used to verify the conversion.
void launch_star_detect_source_gray(const StarDetectSource& source, int height, int width,
                                    double* gray_host);

void bind_star_detect_fused_pixel_components_ops(py::module_& m);
