/* Copyright (c) by CryptoLab Inc. and Seoul National University R&DB Foundation.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */

#include "DeviceVector.h"

#ifdef __APPLE__
#include "MetalHelper.h"

namespace ckks {

DeviceVector::DeviceVector(int size) : size_(size), capacity_(size) {
  if (size > 0) {
    MetalHelper& helper = MetalHelper::getInstance();
    buffer_ = helper.createBuffer(size * sizeof(Dtype));
  } else {
    buffer_ = nil;
  }
}

DeviceVector::DeviceVector(const DeviceVector& ref) : size_(ref.size_), capacity_(ref.size_) {
  if (size_ > 0) {
    MetalHelper& helper = MetalHelper::getInstance();
    buffer_ = helper.createBuffer(size_ * sizeof(Dtype));
    
    auto commandBuffer = [helper.getCommandQueue() commandBuffer];
    auto blitEncoder = [commandBuffer blitCommandEncoder];
    
    [blitEncoder copyFromBuffer:ref.buffer_ 
                   sourceOffset:0
                       toBuffer:buffer_
              destinationOffset:0
                           size:size_ * sizeof(Dtype)];
    
    [blitEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
  } else {
    buffer_ = nil;
  }
}

DeviceVector::DeviceVector(DeviceVector&& other) : buffer_(other.buffer_), size_(other.size_), capacity_(other.capacity_) {
  other.buffer_ = nil;
  other.size_ = 0;
  other.capacity_ = 0;
}

DeviceVector& DeviceVector::operator=(DeviceVector&& other) {
  if (this != &other) {
    buffer_ = other.buffer_;
    size_ = other.size_;
    capacity_ = other.capacity_;
    
    other.buffer_ = nil;
    other.size_ = 0;
    other.capacity_ = 0;
  }
  return *this;
}

DeviceVector::DeviceVector(const HostVector& ref) : size_(ref.size()), capacity_(ref.size()) {
  if (size_ > 0) {
    MetalHelper& helper = MetalHelper::getInstance();
    buffer_ = helper.createBufferWithData(ref.data(), size_ * sizeof(Dtype));
  } else {
    buffer_ = nil;
  }
}

DeviceVector::operator HostVector() const {
  HostVector host(size_);
  if (size_ > 0) {
    MetalHelper& helper = MetalHelper::getInstance();
    helper.copyFromBuffer(host.data(), buffer_, size_ * sizeof(Dtype));
  }
  return host;
}

void DeviceVector::setConstant(const Dtype c) {
  if (size_ > 0 && buffer_) {
    Dtype* ptr = static_cast<Dtype*>(buffer_.contents);
    for (size_t i = 0; i < size_; ++i) {
      ptr[i] = c;
    }
  }
}

void DeviceVector::resize(int new_size) {
  if (new_size <= 0) {
    buffer_ = nil;
    size_ = 0;
    capacity_ = 0;
    return;
  }
  
  if (new_size <= capacity_) {
    size_ = new_size;
    return;
  }
  
  // Need to reallocate
  MetalHelper& helper = MetalHelper::getInstance();
  auto new_buffer = helper.createBuffer(new_size * sizeof(Dtype));
  
  if (buffer_ && size_ > 0) {
    // Copy existing data
    auto commandBuffer = [helper.getCommandQueue() commandBuffer];
    auto blitEncoder = [commandBuffer blitCommandEncoder];
    
    [blitEncoder copyFromBuffer:buffer_ 
                   sourceOffset:0
                       toBuffer:new_buffer
              destinationOffset:0
                           size:size_ * sizeof(Dtype)];
    
    [blitEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
  }
  
  buffer_ = new_buffer;
  size_ = new_size;
  capacity_ = new_size;
}

bool DeviceVector::operator==(const DeviceVector& other) const {
  return HostVector(*this) == HostVector(other);
}

bool DeviceVector::operator!=(const DeviceVector& other) const {
  return !operator==(other);
}

void DeviceVector::append(const DeviceVector& out) {
  size_t old_size = size_;
  resize(size_ + out.size_);
  
  if (out.size_ > 0) {
    MetalHelper& helper = MetalHelper::getInstance();
    
    auto commandBuffer = [helper.getCommandQueue() commandBuffer];
    auto blitEncoder = [commandBuffer blitCommandEncoder];
    
    [blitEncoder copyFromBuffer:out.buffer_ 
                   sourceOffset:0
                       toBuffer:buffer_
              destinationOffset:old_size * sizeof(Dtype)
                           size:out.size_ * sizeof(Dtype)];
    
    [blitEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
  }
}

}  // namespace ckks

#else

// CUDA implementation
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

namespace ckks {

void DeviceVector::append(const DeviceVector& out) {
  size_t old_size = size();
  resize(size() + out.size());
  cudaMemcpyAsync(data() + old_size, out.data(), out.size() * sizeof(Dtype),
                  cudaMemcpyDefault, stream_);
}

}  // namespace ckks

#endif