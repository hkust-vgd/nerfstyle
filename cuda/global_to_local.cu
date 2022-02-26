#include <torch/extension.h>
#include <ATen/cuda/CUDABlas.h>

using namespace torch::indexing;

template <typename scalar_t>
__global__ void global_to_local_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> mid_points,
    const float* voxel_size,
    const int64_t* bsizes,
    const int64_t* offsets
  ) {
  int net_idx = blockIdx.x;
  int point_idx = threadIdx.x;
  int coord_idx = threadIdx.y;
  
  int offset = offsets[net_idx];
  while (point_idx < bsizes[net_idx]) {
    points[offset + point_idx][coord_idx] -= mid_points[net_idx][coord_idx];
    points[offset + point_idx][coord_idx] /= (voxel_size[coord_idx] / 2.0);
    point_idx += blockDim.x;
  }
}

void global_to_local_cuda(
    torch::Tensor& points_tensor,
    const torch::Tensor& mid_points_tensor,
    const torch::Tensor& voxel_size_tensor,
    const torch::Tensor& bsizes_tensor) {
  int num_nets = bsizes_tensor.size(0);
  const int threads = 64;

  auto cumsum_tensor = at::cumsum(bsizes_tensor, 0);
  auto offsets_tensor = torch::zeros_like(cumsum_tensor);
  offsets_tensor.index_put_({Slice(1, num_nets)}, cumsum_tensor.index({Slice(0, num_nets - 1)}));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(points_tensor.type(), "global_to_local", ([&] {
    global_to_local_cuda_kernel<scalar_t><<<num_nets, dim3(threads, 3), 0, stream>>>(
      points_tensor.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
      mid_points_tensor.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
      voxel_size_tensor.data_ptr<float>(),
      bsizes_tensor.data_ptr<int64_t>(),
      offsets_tensor.data_ptr<int64_t>()
    );
  }));
}
