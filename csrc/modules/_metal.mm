#include "common/compat.h"
#include "common/metal_error.h"
#include "common/metal_runtime.h"
#include "ops/metal/filter/median_filter_ops.h"
#include "ops/metal/star_shrink/star_mask_ops.h"
#include "ops/metal/star_shrink/star_shrink_dog_process_ops.h"
#include "ops/metal/star_shrink/star_shrink_process_ops.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_metal, m) {
    m.doc() = "Optional Metal ops for HoshinoWeaver";

    py::register_exception<hnw::MetalRuntimeUnavailableError>(m, "MetalRuntimeUnavailableError",
                                                              PyExc_RuntimeError);
    py::register_exception<hnw::MetalResourceExhaustedError>(m, "MetalResourceExhaustedError",
                                                             PyExc_RuntimeError);

    bind_metal_runtime(m);
    bind_median_filter_metal_ops(m);
    bind_star_shrink_process_metal_ops(m);
    bind_star_mask_dog_metal_ops(m);
    bind_star_shrink_dog_process_metal_ops(m);
}
