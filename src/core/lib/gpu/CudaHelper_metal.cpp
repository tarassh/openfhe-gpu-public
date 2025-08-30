/* Copyright (c) by CryptoLab Inc. and Seoul National University R&DB Foundation.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */
#include "Define.h"

#ifdef __APPLE__
#include "MetalHelper.h"

namespace ckks {

void CudaNvtxStart(std::string msg) { 
    // Metal doesn't have direct NVTX equivalent, could use os_signpost on macOS
    // For now, just a no-op or simple logging
    if (!msg.empty()) {
        std::cout << "[Metal Trace Start]: " << msg << std::endl;
    }
}

void CudaNvtxStop() { 
    // Metal equivalent - no-op for now
    std::cout << "[Metal Trace Stop]" << std::endl;
}

void CudaHostSync() { 
    // Metal equivalent - wait for all command buffers to complete
    MetalHelper& helper = MetalHelper::getInstance();
    auto commandBuffer = [helper.getCommandQueue() commandBuffer];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

}

#else

#include "nvToolsExt.h"

namespace ckks {

void CudaNvtxStart(std::string msg) { nvtxRangePushA(msg.c_str()); }
void CudaNvtxStop() { nvtxRangePop(); }
void CudaHostSync() { cudaDeviceSynchronize(); }

}

#endif