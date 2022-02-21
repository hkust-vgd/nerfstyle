#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void global_to_local_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> mid_points) {
  int idx = threadIdx.x;
}

void global_to_local_cuda(
    const torch::Tensor& points_tensor,
    const torch::Tensor& mid_points_tensor,
    const torch::Tensor& batch_size_per_network_tensor) {
  int blocks = batch_size_per_network_tensor.size(0);
  const int threads = 64;

  AT_DISPATCH_FLOATING_TYPES(points_tensor.type(), "global_to_local", ([&] {
      global_to_local_cuda_kernel<scalar_t><<<blocks, threads>>>(
          points_tensor.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
          mid_points_tensor.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>());
  }));
}
