#include <metal_stdlib>

using namespace metal;

struct MedianStackParams {
    uint n_frames;
    uint plane_size;
    uint top_shift;
};

template <typename T>
uint median_stack_value(device const T* source, constant MedianStackParams& params, uint gid) {
    uint prefix = 0;
    uint rank = params.n_frames / 2 + 1;
    for (int shift = int(params.top_shift); shift >= 0; shift -= 4) {
        ushort buckets[16];
        for (int bin = 0; bin < 16; ++bin) {
            buckets[bin] = 0;
        }
        for (uint frame = 0; frame < params.n_frames; ++frame) {
            const uint value = uint(source[frame * params.plane_size + gid]);
            if (shift == int(params.top_shift) || (value >> uint(shift + 4)) == prefix) {
                ++buckets[(value >> uint(shift)) & 15u];
            }
        }
        for (uint bin = 0; bin < 16; ++bin) {
            if (rank > uint(buckets[bin])) {
                rank -= uint(buckets[bin]);
            } else {
                prefix = (prefix << 4u) | bin;
                break;
            }
        }
    }

    if ((params.n_frames & 1u) != 0) {
        return prefix;
    }
    uint below = 0;
    uint low = 0;
    for (uint frame = 0; frame < params.n_frames; ++frame) {
        const uint value = uint(source[frame * params.plane_size + gid]);
        if (value < prefix) {
            ++below;
            low = max(low, value);
        }
    }
    if (below < params.n_frames / 2) {
        low = prefix;
    }
    return (low + prefix) / 2u;
}

kernel void median_reduce_stack_u8(device const uchar* source [[buffer(0)]],
                                   device uchar* result [[buffer(1)]],
                                   constant MedianStackParams& params [[buffer(2)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid < params.plane_size) {
        result[gid] = uchar(median_stack_value(source, params, gid));
    }
}

kernel void median_reduce_stack_u16(device const ushort* source [[buffer(0)]],
                                    device ushort* result [[buffer(1)]],
                                    constant MedianStackParams& params [[buffer(2)]],
                                    uint gid [[thread_position_in_grid]]) {
    if (gid < params.plane_size) {
        result[gid] = ushort(median_stack_value(source, params, gid));
    }
}
