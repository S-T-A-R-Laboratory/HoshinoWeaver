#include <metal_stdlib>

using namespace metal;

struct MedianStackParams {
    uint n_frames;
    uint plane_size;
    uint top_shift;
    uint channels;
    uint skip_zero_rgb;
};

template <typename T>
bool median_stack_valid(device const T* source, constant MedianStackParams& params, uint frame,
                        uint gid) {
    if (params.skip_zero_rgb == 0) {
        return true;
    }
    const uint rgb = frame * params.plane_size + (gid / params.channels) * params.channels;
    return source[rgb] != T(0) || source[rgb + 1] != T(0) || source[rgb + 2] != T(0);
}

template <typename T>
uint median_stack_value(device const T* source, constant MedianStackParams& params, uint gid) {
    uint valid_count = params.n_frames;
    if (params.skip_zero_rgb != 0) {
        valid_count = 0;
        for (uint frame = 0; frame < params.n_frames; ++frame) {
            valid_count += uint(median_stack_valid(source, params, frame, gid));
        }
    }
    if (valid_count == 0) {
        return 0;
    }
    uint prefix = 0;
    uint rank = valid_count / 2 + 1;
    for (int shift = int(params.top_shift); shift >= 0; shift -= 4) {
        ushort buckets[16];
        for (int bin = 0; bin < 16; ++bin) {
            buckets[bin] = 0;
        }
        for (uint frame = 0; frame < params.n_frames; ++frame) {
            if (!median_stack_valid(source, params, frame, gid)) {
                continue;
            }
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

    if ((valid_count & 1u) != 0) {
        return prefix;
    }
    uint below = 0;
    uint low = 0;
    for (uint frame = 0; frame < params.n_frames; ++frame) {
        if (!median_stack_valid(source, params, frame, gid)) {
            continue;
        }
        const uint value = uint(source[frame * params.plane_size + gid]);
        if (value < prefix) {
            ++below;
            low = max(low, value);
        }
    }
    if (below < valid_count / 2) {
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
