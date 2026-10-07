#include "common/host_parallel_copy.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace hnw {

// Host memory bandwidth saturates with a few copy threads; a full OpenMP team
// only adds fork/join cost (and straggler stalls on busy hosts).
constexpr int kMaxCopyThreads = 8;

void parallel_copy(void* destination, const void* source, const size_t bytes) {
    constexpr size_t block = size_t{1} << 20;
    const int64_t blocks = static_cast<int64_t>((bytes + block - 1) / block);
    auto* out = static_cast<unsigned char*>(destination);
    const auto* in = static_cast<const unsigned char*>(source);
    if (blocks <= 1) {
        std::memcpy(out, in, bytes);
        return;
    }
#if defined(_OPENMP)
    const int threads = static_cast<int>(
        std::min<int64_t>(blocks, std::min(omp_get_max_threads(), kMaxCopyThreads)));
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int64_t b = 0; b < blocks; ++b) {
        const size_t offset = static_cast<size_t>(b) * block;
        std::memcpy(out + offset, in + offset, std::min(block, bytes - offset));
    }
}

} // namespace hnw
