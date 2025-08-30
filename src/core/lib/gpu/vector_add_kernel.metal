#include <metal_stdlib>
using namespace metal;

// Metal compute shader for vector addition
kernel void vectorAdd(device const int* a [[buffer(0)]],
                     device const int* b [[buffer(1)]],
                     device int* c [[buffer(2)]],
                     constant uint& N [[buffer(3)]],
                     uint tid [[thread_position_in_grid]]) {
    // Boundary check
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}