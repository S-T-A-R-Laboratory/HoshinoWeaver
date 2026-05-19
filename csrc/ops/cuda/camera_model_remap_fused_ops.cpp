#include "camera_model_remap_fused_ops.h"

#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>

void launch_camera_model_remap_fused_u8(
    const unsigned char* image_host,
    unsigned char* out_host,
    int src_height,
    int src_width,
    int channels,
    int out_height,
    int out_width,
    float fx_src,
    float fy_src,
    float cx_src,
    float cy_src,
    float fx_dst,
    float fy_dst,
    float cx_dst,
    float cy_dst,
    const float* rotation_dst_to_src);

void launch_camera_model_remap_fused_u16(
    const unsigned short* image_host,
    unsigned short* out_host,
    int src_height,
    int src_width,
    int channels,
    int out_height,
    int out_width,
    float fx_src,
    float fy_src,
    float cx_src,
    float cy_src,
    float fx_dst,
    float fy_dst,
    float cx_dst,
    float cy_dst,
    const float* rotation_dst_to_src);

void launch_camera_model_remap_fused_f32(
    const float* image_host,
    float* out_host,
    int src_height,
    int src_width,
    int channels,
    int out_height,
    int out_width,
    float fx_src,
    float fy_src,
    float cx_src,
    float cy_src,
    float fx_dst,
    float fy_dst,
    float cx_dst,
    float cy_dst,
    const float* rotation_dst_to_src);

namespace {

template <typename T>
using launch_fn_t = void (*)(
    const T* image_host,
    T* out_host,
    int src_height,
    int src_width,
    int channels,
    int out_height,
    int out_width,
    float fx_src,
    float fy_src,
    float cx_src,
    float cy_src,
    float fx_dst,
    float fy_dst,
    float cx_dst,
    float cy_dst,
    const float* rotation_dst_to_src);

template <typename T>
py::array_t<T> camera_model_remap_fused_impl(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& image,
    const ssize_t out_height,
    const ssize_t out_width,
    const float fx_src,
    const float fy_src,
    const float cx_src,
    const float cy_src,
    const float fx_dst,
    const float fy_dst,
    const float cx_dst,
    const float cy_dst,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        rotation_dst_to_src,
    launch_fn_t<T> launcher) {
    if (out_height <= 0 || out_width <= 0) {
        throw std::invalid_argument(
            "camera_model_remap: output height and width must be positive");
    }
    if (image.ndim() != 2 && image.ndim() != 3) {
        throw std::invalid_argument(
            "camera_model_remap: image must have shape (H, W) or (H, W, C)");
    }
    if (rotation_dst_to_src.ndim() != 2 || rotation_dst_to_src.shape(0) != 3 ||
        rotation_dst_to_src.shape(1) != 3) {
        throw std::invalid_argument(
            "camera_model_remap: rotation_dst_to_src must have shape (3, 3)");
    }

    auto image_info = image.request();
    auto rotation_info = rotation_dst_to_src.request();
    const int src_height = static_cast<int>(image.shape(0));
    const int src_width = static_cast<int>(image.shape(1));
    const int channels = image.ndim() == 3 ? static_cast<int>(image.shape(2)) : 1;

    std::vector<py::ssize_t> out_shape = {out_height, out_width};
    if (image.ndim() == 3) {
        out_shape.push_back(static_cast<py::ssize_t>(channels));
    }
    py::array_t<T> out(out_shape);
    auto out_info = out.request();

    {
        py::gil_scoped_release release;
        launcher(
            static_cast<const T*>(image_info.ptr),
            static_cast<T*>(out_info.ptr),
            src_height,
            src_width,
            channels,
            static_cast<int>(out_height),
            static_cast<int>(out_width),
            fx_src,
            fy_src,
            cx_src,
            cy_src,
            fx_dst,
            fy_dst,
            cx_dst,
            cy_dst,
            static_cast<const float*>(rotation_info.ptr));
    }
    return out;
}

py::array camera_model_remap_fused_dispatch(
    const py::array& image,
    const ssize_t out_height,
    const ssize_t out_width,
    const float fx_src,
    const float fy_src,
    const float cx_src,
    const float cy_src,
    const float fx_dst,
    const float fy_dst,
    const float cx_dst,
    const float cy_dst,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        rotation_dst_to_src) {
    if (py::isinstance<py::array_t<unsigned char>>(image)) {
        return camera_model_remap_fused_impl<unsigned char>(
            image.cast<py::array_t<unsigned char>>(),
            out_height,
            out_width,
            fx_src,
            fy_src,
            cx_src,
            cy_src,
            fx_dst,
            fy_dst,
            cx_dst,
            cy_dst,
            rotation_dst_to_src,
            launch_camera_model_remap_fused_u8);
    }
    if (py::isinstance<py::array_t<unsigned short>>(image)) {
        return camera_model_remap_fused_impl<unsigned short>(
            image.cast<py::array_t<unsigned short>>(),
            out_height,
            out_width,
            fx_src,
            fy_src,
            cx_src,
            cy_src,
            fx_dst,
            fy_dst,
            cx_dst,
            cy_dst,
            rotation_dst_to_src,
            launch_camera_model_remap_fused_u16);
    }
    if (py::isinstance<py::array_t<float>>(image)) {
        return camera_model_remap_fused_impl<float>(
            image.cast<py::array_t<float>>(),
            out_height,
            out_width,
            fx_src,
            fy_src,
            cx_src,
            cy_src,
            fx_dst,
            fy_dst,
            cx_dst,
            cy_dst,
            rotation_dst_to_src,
            launch_camera_model_remap_fused_f32);
    }
    throw std::invalid_argument(
        "camera_model_remap: unsupported image dtype; expected uint8/uint16/float32");
}

}  // namespace

void bind_camera_model_remap_fused_ops(py::module_& m) {
    m.def("camera_model_remap",
          &camera_model_remap_fused_dispatch,
          py::arg("image"),
          py::arg("out_height"),
          py::arg("out_width"),
          py::arg("fx_src"),
          py::arg("fy_src"),
          py::arg("cx_src"),
          py::arg("cy_src"),
          py::arg("fx_dst"),
          py::arg("fy_dst"),
          py::arg("cx_dst"),
          py::arg("cy_dst"),
          py::arg("rotation_dst_to_src"),
          "Apply a zero-distortion camera-model remap with fused grid generation and bilinear sampling using CUDA.");
}
