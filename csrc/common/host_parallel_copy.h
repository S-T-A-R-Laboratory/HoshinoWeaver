#pragma once

#include "common/compat.h"

#include <cstddef>

namespace hnw {

// memcpy split across OpenMP threads. Staging large images into pinned memory
// with one thread is bound by a single core's copy bandwidth, which on
// multi-socket hosts is several times below the PCIe transfer rate.
void parallel_copy(void* destination, const void* source, size_t bytes);

} // namespace hnw
