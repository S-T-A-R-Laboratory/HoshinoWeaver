#include <metal_stdlib>

using namespace metal;

struct MedianParams {
    uint height;
    uint width;
};

// Four 4-bit radix passes select rank 85 from the 13x13 nearest-border window.
// Small thread-local histograms keep the result exact for all uint16 inputs.
kernel void median_filter_u16_13(device const ushort* source [[buffer(0)]],
                                 device ushort* result [[buffer(1)]],
                                 constant MedianParams& params [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= params.height * params.width) {
        return;
    }
    const int center_y = int(gid / params.width);
    const int center_x = int(gid % params.width);
    const int height = int(params.height);
    const int width = int(params.width);
    uint prefix = 0;
    uint rank = 85;
    for (int shift = 12; shift >= 0; shift -= 4) {
        ushort buckets[16];
        for (int bin = 0; bin < 16; ++bin) {
            buckets[bin] = 0;
        }
        for (int dy = -6; dy <= 6; ++dy) {
            const uint y = uint(clamp(center_y + dy, 0, height - 1));
            for (int dx = -6; dx <= 6; ++dx) {
                const uint x = uint(clamp(center_x + dx, 0, width - 1));
                const uint value = uint(source[y * params.width + x]);
                if (shift == 12 || (value >> uint(shift + 4)) == prefix) {
                    ++buckets[(value >> uint(shift)) & 15u];
                }
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
    result[gid] = ushort(prefix);
}
