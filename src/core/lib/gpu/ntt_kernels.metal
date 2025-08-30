#include <metal_stdlib>
using namespace metal;

// Basic modular arithmetic helpers for Metal
device inline uint64_t mul_and_reduce_shoup_metal(uint64_t a, uint64_t b, uint64_t b_shoup, uint64_t p) {
    // Simplified implementation - in production would need proper Barrett reduction
    return (a * b) % p;
}

// NTT butterfly operation (Metal version)
device inline void butt_ntt_local_metal(thread uint64_t &a, thread uint64_t &b,
                                        const uint64_t w, const uint64_t w_,
                                        const uint64_t p) {
    uint64_t two_p = 2 * p;
    uint64_t U = mul_and_reduce_shoup_metal(b, w, w_, p);
    if (a >= two_p) a -= two_p;
    b = a + (two_p - U);
    a += U;
}

// Inverse NTT butterfly operation (Metal version)  
device inline void butt_intt_local_metal(thread uint64_t &x, thread uint64_t &y,
                                         const uint64_t w, const uint64_t w_,
                                         const uint64_t p) {
    const uint64_t two_p = 2 * p;
    const uint64_t T = two_p - y + x;
    uint64_t new_x = x + y;
    if (new_x >= two_p) new_x -= two_p;
    if (T & 1) new_x += p;
    x = (new_x >> 1);
    y = mul_and_reduce_shoup_metal(T, w, w_, p);
}

// Simple NTT kernel (8-point per thread, Phase 1, Out-of-Place)
kernel void Intt8PointPerThreadPhase1OoP_metal(
    device const uint64_t* input [[buffer(0)]],
    device uint64_t* output [[buffer(1)]],
    device const uint64_t* base_inv [[buffer(2)]],
    device const uint64_t* base_inv_ [[buffer(3)]],
    device const uint64_t* primes [[buffer(4)]],
    constant uint& m [[buffer(5)]],
    constant uint& num_prime [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& start_prime_idx [[buffer(8)]],
    constant uint& pad [[buffer(9)]],
    constant uint& radix [[buffer(10)]],
    uint thread_id [[thread_position_in_grid]],
    uint thread_in_group [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]) {
    
    if (thread_id >= (N / 8 * num_prime)) return;
    
    threadgroup uint64_t temp[2048]; // Shared memory equivalent
    
    uint64_t local[8];
    int t = N / 2 / m;
    
    // Prime index
    int np_idx = thread_id / (N / 8) + start_prime_idx;
    // Index in N/2 range
    int N_idx = thread_id % (N / 8);
    // i'th block
    int m_idx = N_idx / (t / 4);
    int t_idx = N_idx % (t / 4);
    
    // Base addresses
    device const uint64_t* in_addr = input + np_idx * N;
    device uint64_t* out_addr = output + np_idx * N;
    device const uint64_t* prime_table = primes;
    device const uint64_t* WInv = base_inv + N * np_idx;
    device const uint64_t* WInv_ = base_inv_ + N * np_idx;
    uint64_t prime = prime_table[np_idx];
    
    int Warp_t = thread_in_group % pad;
    int WarpID = thread_in_group / pad;
    int N_init = 2 * t / radix * WarpID + Warp_t + pad * (t_idx / (radix * pad));
    
    // Load 8 elements
    for (int j = 0; j < 8; j++) {
        local[j] = in_addr[N_init + t / 4 / radix * j];
    }
    
    int tw_idx = m + m_idx;
    int tw_idx2 = radix * tw_idx + WarpID;
    
    // Butterfly operations (simplified)
    for (int j = 0; j < 4; j++) {
        butt_intt_local_metal(local[2 * j], local[2 * j + 1], 
                             WInv[4 * tw_idx2 + j], WInv_[4 * tw_idx2 + j], prime);
    }
    
    for (int j = 0; j < 2; j++) {
        butt_intt_local_metal(local[4 * j], local[4 * j + 2], 
                             WInv[2 * tw_idx2 + j], WInv_[2 * tw_idx2 + j], prime);
        butt_intt_local_metal(local[4 * j + 1], local[4 * j + 3], 
                             WInv[2 * tw_idx2 + j], WInv_[2 * tw_idx2 + j], prime);
    }
    
    butt_intt_local_metal(local[0], local[4], WInv[tw_idx2], WInv_[tw_idx2], prime);
    butt_intt_local_metal(local[1], local[5], WInv[tw_idx2], WInv_[tw_idx2], prime);
    butt_intt_local_metal(local[2], local[6], WInv[tw_idx2], WInv_[tw_idx2], prime);
    butt_intt_local_metal(local[3], local[7], WInv[tw_idx2], WInv_[tw_idx2], prime);
    
    // Store results
    for (int j = 0; j < 8; j++) {
        out_addr[N_init + t / 4 / radix * j] = local[j];
    }
}

// Simple NTT kernel (8-point per thread, Phase 1, Forward)
kernel void Ntt8PointPerThreadPhase1_metal(
    device const uint64_t* input [[buffer(0)]],
    device uint64_t* output [[buffer(1)]],
    device const uint64_t* twiddles [[buffer(2)]],
    device const uint64_t* twiddles_ [[buffer(3)]],
    device const uint64_t* primes [[buffer(4)]],
    constant uint& m [[buffer(5)]],
    constant uint& num_prime [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& start_prime_idx [[buffer(8)]],
    uint thread_id [[thread_position_in_grid]],
    uint thread_in_group [[thread_position_in_threadgroup]]) {
    
    if (thread_id >= (N / 8 * num_prime)) return;
    
    uint64_t local[8];
    int t = N / 2 / m;
    
    // Prime index
    int np_idx = thread_id / (N / 8) + start_prime_idx;
    // Index in N/2 range
    int N_idx = thread_id % (N / 8);
    // i'th block
    int m_idx = N_idx / (t / 4);
    int t_idx = N_idx % (t / 4);
    
    // Base addresses
    device const uint64_t* in_addr = input + np_idx * N;
    device uint64_t* out_addr = output + np_idx * N;
    device const uint64_t* prime_table = primes;
    device const uint64_t* W = twiddles + N * np_idx;
    device const uint64_t* W_ = twiddles_ + N * np_idx;
    uint64_t prime = prime_table[np_idx];
    
    // Load 8 elements
    for (int j = 0; j < 8; j++) {
        local[j] = in_addr[2 * m_idx * t + t_idx + t / 4 * j];
    }
    
    int tw_idx = m + m_idx;
    
    // Forward NTT butterfly operations (simplified)
    butt_ntt_local_metal(local[0], local[4], W[tw_idx], W_[tw_idx], prime);
    butt_ntt_local_metal(local[1], local[5], W[tw_idx], W_[tw_idx], prime);
    butt_ntt_local_metal(local[2], local[6], W[tw_idx], W_[tw_idx], prime);
    butt_ntt_local_metal(local[3], local[7], W[tw_idx], W_[tw_idx], prime);
    
    for (int j = 0; j < 2; j++) {
        butt_ntt_local_metal(local[4 * j], local[4 * j + 2], 
                            W[2 * tw_idx + j], W_[2 * tw_idx + j], prime);
        butt_ntt_local_metal(local[4 * j + 1], local[4 * j + 3], 
                            W[2 * tw_idx + j], W_[2 * tw_idx + j], prime);
    }
    
    for (int j = 0; j < 4; j++) {
        butt_ntt_local_metal(local[2 * j], local[2 * j + 1], 
                            W[4 * tw_idx + j], W_[4 * tw_idx + j], prime);
    }
    
    // Store results
    for (int j = 0; j < 8; j++) {
        out_addr[2 * m_idx * t + t_idx + t / 4 * j] = local[j];
    }
}

// Modular reduction step
kernel void modUpStepTwoSimple_metal(
    device const uint64_t* ptr_after_intt [[buffer(0)]],
    device const uint64_t* ptr_hat_inv_mod_down [[buffer(1)]],
    device const uint64_t* ptr_hat_inv_mod_down_ [[buffer(2)]],
    device const uint64_t* primes [[buffer(3)]],
    device uint64_t* result [[buffer(4)]],
    constant uint& num_prime [[buffer(5)]],
    constant uint& degree [[buffer(6)]],
    constant uint& down_start [[buffer(7)]],
    uint thread_id [[thread_position_in_grid]]) {
    
    if (thread_id >= num_prime * degree) return;
    
    uint prime_idx = thread_id / degree;
    uint element_idx = thread_id % degree;
    
    uint64_t prime = primes[down_start + prime_idx];
    uint64_t val = ptr_after_intt[thread_id];
    uint64_t hat_inv = ptr_hat_inv_mod_down[prime_idx * degree + element_idx];
    uint64_t hat_inv_ = ptr_hat_inv_mod_down_[prime_idx * degree + element_idx];
    
    // Simplified modular multiplication
    result[thread_id] = mul_and_reduce_shoup_metal(val, hat_inv, hat_inv_, prime);
}