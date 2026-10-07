#include "asterism_mutual_nearest_ops.h"

#include "common/compat.h"

#include <vector>

namespace {

using hnw::asterism::TokenArray;

py::object asterism_mutual_nearest_cuda_impl(const TokenArray& values1, const TokenArray& values2,
                                             const double threshold) {
    hnw::asterism::validate_inputs(values1, values2, threshold);
    const int64_t n1 = values1.shape(0);
    const int64_t n2 = values2.shape(0);
    if (n1 == 0 || n2 == 0) {
        return hnw::asterism::empty_pairs();
    }

    std::vector<int32_t> mutual12;
    bool representable = true;
    {
        py::gil_scoped_release release;
        representable = launch_asterism_mutual_nearest_cuda(values1.data(), n1, values2.data(), n2,
                                                            threshold, &mutual12);
    }
    if (!representable) {
        return py::none();
    }
    return hnw::asterism::mutual_pairs_to_arrays(mutual12);
}

} // namespace

void bind_asterism_mutual_nearest_cuda_ops(py::module_& m) {
    m.def("asterism_mutual_nearest_cuda", &asterism_mutual_nearest_cuda_impl, py::arg("values1"),
          py::arg("values2"), py::arg("threshold"),
          "Mutual nearest asterism-token pairs within a Euclidean threshold using CUDA; "
          "returns None when the token extent cannot be gridded or a nearest distance ties.");
}
