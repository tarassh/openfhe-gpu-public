// This program computes the sum of two vectors of length N
// Metal implementation

#include "vector_add_kernel.h"

#ifdef __APPLE__
#include "MetalHelper.h"

void add_vectors(std::vector<int> &a, std::vector<int> &b,
                   std::vector<int> &c) {
    ckks::MetalHelper& metalHelper = ckks::MetalHelper::getInstance();
    
    const size_t N = a.size();
    const size_t bytes = sizeof(int) * N;
    
    // Create Metal buffers
    auto bufferA = metalHelper.createBufferWithData(a.data(), bytes);
    auto bufferB = metalHelper.createBufferWithData(b.data(), bytes);
    auto bufferC = metalHelper.createBuffer(bytes);
    auto bufferN = metalHelper.createBufferWithData(&N, sizeof(uint32_t));
    
    // Create compute pipeline
    auto pipelineState = metalHelper.createComputePipelineState(@"vectorAdd");
    
    // Execute computation
    NSArray<id<MTLBuffer>>* buffers = @[bufferA, bufferB, bufferC, bufferN];
    MTLSize gridSize = MTLSizeMake(N, 1, 1);
    
    metalHelper.executeCompute(pipelineState, buffers, gridSize);
    
    // Copy result back to host
    metalHelper.copyFromBuffer(c.data(), bufferC, bytes);
}

void verify_result(std::vector<int> &a, std::vector<int> &b,
                   std::vector<int> &c) {
    for (int i = 0; i < a.size(); i++) {
        assert(c[i] == a[i] + b[i]);
    }
}

#else
// Fallback CUDA implementation for non-Apple platforms

// CUDA kernel for vector addition
__global__ void vectorAdd(const int *__restrict a, const int *__restrict b,
                          int *__restrict c, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < N) c[tid] = a[tid] + b[tid];
}

void add_vectors(std::vector<int> &a, std::vector<int> &b,
                   std::vector<int> &c) {
  const size_t N = a.size();
  const size_t bytes = sizeof(int) * N;

  // Allocate memory on the device
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  // Copy data from the host to the device (CPU -> GPU)
  cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

  // Threads per CTA (1024)
  int NUM_THREADS = 1 << 10;
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

  // Launch the kernel on the GPU
  vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);

  // Copy sum vector from device to host
  cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);                 

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

void verify_result(std::vector<int> &a, std::vector<int> &b,
                   std::vector<int> &c) {
  for (int i = 0; i < a.size(); i++) {
    assert(c[i] == a[i] + b[i]);
  }
}

#endif