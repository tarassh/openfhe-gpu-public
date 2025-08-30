#include "MetalHelper.h"

#ifdef __APPLE__

namespace ckks {

MetalHelper& MetalHelper::getInstance() {
    static MetalHelper instance;
    return instance;
}

void MetalHelper::initialize() {
    if (initialized_) return;
    
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) {
        throw std::runtime_error("Metal is not supported on this device");
    }
    
    commandQueue_ = [device_ newCommandQueue];
    if (!commandQueue_) {
        throw std::runtime_error("Failed to create Metal command queue");
    }
    
    loadLibrary();
    
    initialized_ = true;
}

void MetalHelper::loadLibrary() {
    NSError* error = nil;
    
    // Try to load precompiled library first
    NSString* libraryPath = [[NSBundle mainBundle] pathForResource:@"openfhe_metal" ofType:@"metallib"];
    if (libraryPath) {
        library_ = [device_ newLibraryWithFile:libraryPath error:&error];
        if (library_) return;
    }
    
    // If precompiled library not found, compile from source
    NSString* shaderSource = @R"(
#include <metal_stdlib>
using namespace metal;

// Vector addition kernel
kernel void vectorAdd(device const int* a [[buffer(0)]],
                     device const int* b [[buffer(1)]],
                     device int* c [[buffer(2)]],
                     constant uint& N [[buffer(3)]],
                     uint tid [[thread_position_in_grid]]) {
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

// Basic NTT butterfly operation helper
device inline void butt_ntt_local_metal(thread uint64_t &a, thread uint64_t &b,
                                        const uint64_t w, const uint64_t w_,
                                        const uint64_t p) {
    uint64_t two_p = 2 * p;
    // Simplified modular multiplication - in real implementation would need proper reduction
    uint64_t U = (b * w) % p;
    if (a >= two_p) a -= two_p;
    b = a + (two_p - U);
    a += U;
}

// Basic inverse NTT butterfly operation helper  
device inline void butt_intt_local_metal(thread uint64_t &x, thread uint64_t &y,
                                         const uint64_t w, const uint64_t w_,
                                         const uint64_t p) {
    const uint64_t two_p = 2 * p;
    const uint64_t T = two_p - y + x;
    uint64_t new_x = x + y;
    if (new_x >= two_p) new_x -= two_p;
    if (T & 1) new_x += p;
    x = (new_x >> 1);
    // Simplified modular multiplication
    y = (T * w) % p;
}

// Simple NTT kernel (simplified version)
kernel void simpleNTT(device const uint64_t* input [[buffer(0)]],
                     device uint64_t* output [[buffer(1)]],
                     device const uint64_t* twiddles [[buffer(2)]],
                     device const uint64_t* twiddles_ [[buffer(3)]],
                     device const uint64_t* primes [[buffer(4)]],
                     constant uint& N [[buffer(5)]],
                     constant uint& num_primes [[buffer(6)]],
                     uint tid [[thread_position_in_grid]]) {
    if (tid >= N * num_primes) return;
    
    uint prime_idx = tid / N;
    uint element_idx = tid % N;
    uint64_t prime = primes[prime_idx];
    
    // Simple copy for now - full NTT implementation would be more complex
    output[tid] = input[tid];
}
    )";
    
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.fastMathEnabled = YES;
    
    library_ = [device_ newLibraryWithSource:shaderSource options:options error:&error];
    if (!library_) {
        NSLog(@"Error compiling Metal library: %@", error.localizedDescription);
        throw std::runtime_error("Failed to compile Metal shaders");
    }
}

id<MTLBuffer> MetalHelper::createBuffer(size_t size, MTLResourceOptions options) {
    if (!initialized_) initialize();
    
    id<MTLBuffer> buffer = [device_ newBufferWithLength:size options:options];
    if (!buffer) {
        throw std::runtime_error("Failed to create Metal buffer");
    }
    return buffer;
}

id<MTLBuffer> MetalHelper::createBufferWithData(const void* data, size_t size, MTLResourceOptions options) {
    if (!initialized_) initialize();
    
    id<MTLBuffer> buffer = [device_ newBufferWithBytes:data length:size options:options];
    if (!buffer) {
        throw std::runtime_error("Failed to create Metal buffer with data");
    }
    return buffer;
}

id<MTLComputePipelineState> MetalHelper::createComputePipelineState(NSString* functionName) {
    if (!initialized_) initialize();
    
    id<MTLFunction> function = [library_ newFunctionWithName:functionName];
    if (!function) {
        NSLog(@"Function %@ not found in Metal library", functionName);
        throw std::runtime_error("Metal function not found");
    }
    
    NSError* error = nil;
    id<MTLComputePipelineState> pipelineState = [device_ newComputePipelineStateWithFunction:function error:&error];
    if (!pipelineState) {
        NSLog(@"Error creating pipeline state: %@", error.localizedDescription);
        throw std::runtime_error("Failed to create Metal pipeline state");
    }
    
    return pipelineState;
}

void MetalHelper::executeCompute(id<MTLComputePipelineState> pipelineState,
                                NSArray<id<MTLBuffer>>* buffers,
                                MTLSize gridSize,
                                MTLSize threadgroupSize) {
    if (!initialized_) initialize();
    
    if (threadgroupSize.width == 0) {
        threadgroupSize = calculateOptimalThreadgroupSize(pipelineState, gridSize);
    }
    
    id<MTLCommandBuffer> commandBuffer = [commandQueue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipelineState];
    
    for (NSUInteger i = 0; i < buffers.count; i++) {
        [encoder setBuffer:buffers[i] offset:0 atIndex:i];
    }
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    if (commandBuffer.error) {
        NSLog(@"Metal compute error: %@", commandBuffer.error.localizedDescription);
        throw std::runtime_error("Metal compute execution failed");
    }
}

void MetalHelper::copyToBuffer(id<MTLBuffer> buffer, const void* data, size_t size) {
    memcpy(buffer.contents, data, size);
}

void MetalHelper::copyFromBuffer(void* data, id<MTLBuffer> buffer, size_t size) {
    memcpy(data, buffer.contents, size);
}

MTLSize calculateOptimalThreadgroupSize(id<MTLComputePipelineState> pipelineState, MTLSize gridSize) {
    NSUInteger maxThreadsPerGroup = pipelineState.maxTotalThreadsPerThreadgroup;
    NSUInteger threadsPerGroup = std::min(static_cast<NSUInteger>(gridSize.width), maxThreadsPerGroup);
    
    // Find largest power of 2 <= threadsPerGroup for optimal performance
    NSUInteger optimalSize = 1;
    while (optimalSize * 2 <= threadsPerGroup) {
        optimalSize *= 2;
    }
    
    return MTLSizeMake(optimalSize, 1, 1);
}

} // namespace ckks

#endif // __APPLE__