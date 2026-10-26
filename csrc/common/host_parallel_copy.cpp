#include "common/host_parallel_copy.h"

#include "common/compat.h"

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

void parallel_copy_frames(void* destination, const void* const* sources, const size_t frame_bytes,
                          const size_t n_frames) {
    auto* out = static_cast<unsigned char*>(destination);
#if defined(_OPENMP)
    const int threads = static_cast<int>(
        std::min(n_frames, static_cast<size_t>(std::min(omp_get_max_threads(), kMaxCopyThreads))));
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int64_t frame = 0; frame < static_cast<int64_t>(n_frames); ++frame) {
        std::memcpy(out + static_cast<size_t>(frame) * frame_bytes, sources[frame], frame_bytes);
    }
}

} // namespace hnw
