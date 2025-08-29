#if defined(__APPLE__)
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <vector>
#include <cassert>

#include "vector_add_kernel.h"

static NSString* const kVectorAddShaderSrc = @"using namespace metal;\n"
@"kernel void vector_add(const device int* a [[ buffer(0) ]],\n"
@"                      const device int* b [[ buffer(1) ]],\n"
@"                      device int* c [[ buffer(2) ]],\n"
@"                      constant int& n [[ buffer(3) ]],\n"
@"                      uint gid [[ thread_position_in_grid ]]) {\n"
@"  if (gid < static_cast<uint>(n)) { c[gid] = a[gid] + b[gid]; }\n"
@"}\n";

static id<MTLComputePipelineState> BuildPipeline(id<MTLDevice> device) {
  NSError* error = nil;
  MTLCompileOptions* opts = [MTLCompileOptions new];
  id<MTLLibrary> lib = [device newLibraryWithSource:kVectorAddShaderSrc options:opts error:&error];
  if (!lib) {
    NSLog(@"Metal compile error: %@", error);
    return nil;
  }
  id<MTLFunction> fn = [lib newFunctionWithName:@"vector_add"];
  if (!fn) return nil;
  id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
  if (!pso) {
    NSLog(@"Metal pipeline error: %@", error);
  }
  return pso;
}

void add_vectors(std::vector<int> &a, std::vector<int> &b, std::vector<int> &c) {
  const int N = static_cast<int>(a.size());
  c.resize(N);

  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> queue = [device newCommandQueue];
  id<MTLComputePipelineState> pso = BuildPipeline(device);

  const NSUInteger bytes = sizeof(int) * N;
  id<MTLBuffer> bufA = [device newBufferWithBytes:a.data() length:bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufB = [device newBufferWithBytes:b.data() length:bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufC = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufN = [device newBufferWithBytes:&N length:sizeof(int) options:MTLResourceStorageModeShared];

  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pso];
  [enc setBuffer:bufA offset:0 atIndex:0];
  [enc setBuffer:bufB offset:0 atIndex:1];
  [enc setBuffer:bufC offset:0 atIndex:2];
  [enc setBuffer:bufN offset:0 atIndex:3];

  const NSUInteger threadsPerThreadgroup = 256;
  MTLSize tg = MTLSizeMake(threadsPerThreadgroup, 1, 1);
  const NSUInteger numTg = (N + threadsPerThreadgroup - 1) / threadsPerThreadgroup;
  MTLSize grid = MTLSizeMake(numTg * threadsPerThreadgroup, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:tg];
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];

  memcpy(c.data(), [bufC contents], bytes);
}

void verify_result(std::vector<int> &a, std::vector<int> &b, std::vector<int> &c) {
  for (int i = 0; i < static_cast<int>(a.size()); i++) {
    assert(c[i] == a[i] + b[i]);
  }
}

#endif  // __APPLE__


