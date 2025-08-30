#ifndef METAL_HELPER_H
#define METAL_HELPER_H

#ifdef __APPLE__
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#include <Foundation/Foundation.h>
#include <vector>
#include <memory>
#include <stdexcept>

namespace ckks {

class MetalHelper {
public:
    static MetalHelper& getInstance();
    
    // Initialize Metal context
    void initialize();
    
    // Buffer management
    id<MTLBuffer> createBuffer(size_t size, MTLResourceOptions options = MTLResourceStorageModeShared);
    id<MTLBuffer> createBufferWithData(const void* data, size_t size, MTLResourceOptions options = MTLResourceStorageModeShared);
    
    // Compute pipeline management
    id<MTLComputePipelineState> createComputePipelineState(NSString* functionName);
    
    // Command execution
    void executeCompute(id<MTLComputePipelineState> pipelineState, 
                       NSArray<id<MTLBuffer>>* buffers,
                       MTLSize gridSize,
                       MTLSize threadgroupSize = MTLSizeMake(0, 0, 0));
    
    // Device properties
    id<MTLDevice> getDevice() const { return device_; }
    id<MTLCommandQueue> getCommandQueue() const { return commandQueue_; }
    
    // Synchronous memory copy utilities
    void copyToBuffer(id<MTLBuffer> buffer, const void* data, size_t size);
    void copyFromBuffer(void* data, id<MTLBuffer> buffer, size_t size);
    
private:
    MetalHelper() = default;
    ~MetalHelper() = default;
    MetalHelper(const MetalHelper&) = delete;
    MetalHelper& operator=(const MetalHelper&) = delete;
    
    void loadLibrary();
    
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> commandQueue_ = nil;
    id<MTLLibrary> library_ = nil;
    bool initialized_ = false;
};

// Utility functions for converting CUDA-style launches to Metal
MTLSize calculateOptimalThreadgroupSize(id<MTLComputePipelineState> pipelineState, MTLSize gridSize);

} // namespace ckks

#endif // __APPLE__

#endif // METAL_HELPER_H