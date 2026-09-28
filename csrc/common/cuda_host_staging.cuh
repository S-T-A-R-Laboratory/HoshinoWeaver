#pragma once

#include "common/compat.h"
#include "common/cuda_runtime_utils.cuh"
#include "common/host_parallel_copy.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>

namespace hnw::cuda {

// Two pinned slots stage host transfers: a parallel host copy works on one
// slot while the DMA of the other is in flight.
constexpr size_t kStagingSlotBytes = size_t{8} << 20;

// Tracks, per slot, the last DMA that reads or writes it. Host code must wait
// for that DMA before refilling the slot, including across transfers that
// share the slots.
class StagingSlots {
public:
    StagingSlots() {
        for (cudaEvent_t& event : events_) {
            throw_if_failed(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                            "staging cudaEventCreate");
        }
    }
    StagingSlots(const StagingSlots&) = delete;
    StagingSlots& operator=(const StagingSlots&) = delete;
    ~StagingSlots() {
        for (cudaEvent_t event : events_) {
            cudaEventDestroy(event);
        }
    }

    void record(const size_t slot, const cudaStream_t stream) {
        throw_if_failed(cudaEventRecord(events_[slot], stream), "staging cudaEventRecord");
        pending_[slot] = true;
    }

    void wait(const size_t slot) {
        if (pending_[slot]) {
            throw_if_failed(cudaEventSynchronize(events_[slot]), "staging cudaEventSynchronize");
            pending_[slot] = false;
        }
    }

private:
    cudaEvent_t events_[2] = {nullptr, nullptr};
    bool pending_[2] = {false, false};
};

inline void staged_upload(void* device_dst, const void* host_src, const size_t bytes,
                          unsigned char* staging, StagingSlots* slots, const cudaStream_t stream) {
    size_t slot = 0;
    for (size_t offset = 0; offset < bytes; offset += kStagingSlotBytes, slot ^= 1) {
        const size_t count = std::min(kStagingSlotBytes, bytes - offset);
        slots->wait(slot);
        unsigned char* stage = staging + slot * kStagingSlotBytes;
        hnw::parallel_copy(stage, static_cast<const unsigned char*>(host_src) + offset, count);
        throw_if_failed(cudaMemcpyAsync(static_cast<unsigned char*>(device_dst) + offset, stage,
                                        count, cudaMemcpyHostToDevice, stream),
                        "staging upload");
        slots->record(slot, stream);
    }
}

inline void staged_download(void* host_dst, const void* device_src, const size_t bytes,
                            unsigned char* staging, StagingSlots* slots,
                            const cudaStream_t stream) {
    const size_t chunks = (bytes + kStagingSlotBytes - 1) / kStagingSlotBytes;
    const auto issue = [&](const size_t chunk) {
        const size_t offset = chunk * kStagingSlotBytes;
        const size_t slot = chunk & 1;
        throw_if_failed(cudaMemcpyAsync(staging + slot * kStagingSlotBytes,
                                        static_cast<const unsigned char*>(device_src) + offset,
                                        std::min(kStagingSlotBytes, bytes - offset),
                                        cudaMemcpyDeviceToHost, stream),
                        "staging download");
        slots->record(slot, stream);
    };
    if (chunks > 0) {
        issue(0);
    }
    for (size_t chunk = 0; chunk < chunks; ++chunk) {
        // The other slot was drained in the previous iteration.
        if (chunk + 1 < chunks) {
            issue(chunk + 1);
        }
        const size_t offset = chunk * kStagingSlotBytes;
        const size_t slot = chunk & 1;
        slots->wait(slot);
        hnw::parallel_copy(static_cast<unsigned char*>(host_dst) + offset,
                           staging + slot * kStagingSlotBytes,
                           std::min(kStagingSlotBytes, bytes - offset));
    }
}

} // namespace hnw::cuda
