/* Copyright (c) by CryptoLab Inc. and Seoul National University R&DB Foundation.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */

#ifdef __APPLE__

#include "NttImple.h"
#include "MetalHelper.h"

namespace ckks {

void Intt8PointPerThreadPhase1OoP_metal(
    const DeviceVector& input, DeviceVector& output,
    const DeviceVector& base_inv, const DeviceVector& base_inv_,
    const DeviceVector& primes, int m, int num_prime, int N,
    int start_prime_idx, int pad, int radix) {
    
    MetalHelper& helper = MetalHelper::getInstance();
    
    // Create compute pipeline
    auto pipelineState = helper.createComputePipelineState(@"Intt8PointPerThreadPhase1OoP_metal");
    
    // Create parameter buffers
    auto m_buffer = helper.createBufferWithData(&m, sizeof(uint32_t));
    auto num_prime_buffer = helper.createBufferWithData(&num_prime, sizeof(uint32_t));
    auto N_buffer = helper.createBufferWithData(&N, sizeof(uint32_t));
    auto start_prime_idx_buffer = helper.createBufferWithData(&start_prime_idx, sizeof(uint32_t));
    auto pad_buffer = helper.createBufferWithData(&pad, sizeof(uint32_t));
    auto radix_buffer = helper.createBufferWithData(&radix, sizeof(uint32_t));
    
    // Set up buffers array
    NSArray<id<MTLBuffer>>* buffers = @[
        input.getBuffer(), output.getBuffer(),
        base_inv.getBuffer(), base_inv_.getBuffer(), primes.getBuffer(),
        m_buffer, num_prime_buffer, N_buffer,
        start_prime_idx_buffer, pad_buffer, radix_buffer
    ];
    
    // Calculate grid size
    uint32_t total_threads = (N / 8) * num_prime;
    MTLSize gridSize = MTLSizeMake(total_threads, 1, 1);
    
    // Execute compute shader
    helper.executeCompute(pipelineState, buffers, gridSize);
}

void Ntt8PointPerThreadPhase1_metal(
    DeviceVector& op, int m, int num_prime, int N,
    int start_prime_idx, int pad, int radix,
    const DeviceVector& base_inv, const DeviceVector& base_inv_,
    const DeviceVector& primes) {
    
    MetalHelper& helper = MetalHelper::getInstance();
    
    // Create compute pipeline
    auto pipelineState = helper.createComputePipelineState(@"Ntt8PointPerThreadPhase1_metal");
    
    // Create parameter buffers
    auto m_buffer = helper.createBufferWithData(&m, sizeof(uint32_t));
    auto num_prime_buffer = helper.createBufferWithData(&num_prime, sizeof(uint32_t));
    auto N_buffer = helper.createBufferWithData(&N, sizeof(uint32_t));
    auto start_prime_idx_buffer = helper.createBufferWithData(&start_prime_idx, sizeof(uint32_t));
    
    // Set up buffers array - input and output are the same for in-place operation
    NSArray<id<MTLBuffer>>* buffers = @[
        op.getBuffer(), op.getBuffer(),  // in-place operation
        base_inv.getBuffer(), base_inv_.getBuffer(), primes.getBuffer(),
        m_buffer, num_prime_buffer, N_buffer, start_prime_idx_buffer
    ];
    
    // Calculate grid size
    uint32_t total_threads = (N / 8) * num_prime;
    MTLSize gridSize = MTLSizeMake(total_threads, 1, 1);
    
    // Execute compute shader
    helper.executeCompute(pipelineState, buffers, gridSize);
}

void modUpStepTwoSimple_metal(
    const DeviceVector& ptr_after_intt, const DeviceVector& ptr_hat_inv_mod_down,
    const DeviceVector& ptr_hat_inv_mod_down_, const DeviceVector& primes,
    DeviceVector& result, int num_prime, int degree, int down_start) {
    
    MetalHelper& helper = MetalHelper::getInstance();
    
    // Create compute pipeline
    auto pipelineState = helper.createComputePipelineState(@"modUpStepTwoSimple_metal");
    
    // Create parameter buffers
    auto num_prime_buffer = helper.createBufferWithData(&num_prime, sizeof(uint32_t));
    auto degree_buffer = helper.createBufferWithData(&degree, sizeof(uint32_t));
    auto down_start_buffer = helper.createBufferWithData(&down_start, sizeof(uint32_t));
    
    // Set up buffers array
    NSArray<id<MTLBuffer>>* buffers = @[
        ptr_after_intt.getBuffer(),
        ptr_hat_inv_mod_down.getBuffer(),
        ptr_hat_inv_mod_down_.getBuffer(),
        primes.getBuffer(),
        result.getBuffer(),
        num_prime_buffer, degree_buffer, down_start_buffer
    ];
    
    // Calculate grid size
    uint32_t total_threads = num_prime * degree;
    MTLSize gridSize = MTLSizeMake(total_threads, 1, 1);
    
    // Execute compute shader
    helper.executeCompute(pipelineState, buffers, gridSize);
}

}  // namespace ckks

#endif // __APPLE__