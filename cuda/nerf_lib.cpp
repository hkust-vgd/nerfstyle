#include <torch/extension.h>
#include "streams.cuh"
#include "multimatmul.cuh"

// CUDA forward declarations

void global_to_local_cuda(
    torch::Tensor& points_tensor,
    const torch::Tensor& mid_points_tensor,
    const torch::Tensor& voxel_size_tensor,
    const torch::Tensor& bsizes_tensor);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CPU(x) AT_ASSERTM(!x.type().is_cuda(), #x " must be a CPU tensor")
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

#define CHECK_CPU_INPUT(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)
#define CHECK_CUDA_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void global_to_local(
    torch::Tensor& points_tensor,
    const torch::Tensor& mid_points_tensor,
    const torch::Tensor& voxel_size_tensor,
    const torch::Tensor& bsizes_tensor) {
  CHECK_CUDA_INPUT(points_tensor);
  CHECK_CUDA_INPUT(mid_points_tensor);
  CHECK_CUDA_INPUT(voxel_size_tensor);
  CHECK_CUDA_INPUT(bsizes_tensor);

  global_to_local_cuda(points_tensor, mid_points_tensor, voxel_size_tensor, bsizes_tensor);
}

torch::Tensor multimatmul_forward(
    const torch::Tensor& weights_tensor,
    const torch::Tensor& biases_tensor,
    const torch::Tensor& inputs_tensor,
    const torch::Tensor& bsizes_tensor,
    int aux_index) {
  CHECK_CUDA_INPUT(weights_tensor);
  CHECK_CUDA_INPUT(biases_tensor);
  CHECK_CUDA_INPUT(inputs_tensor);
  CHECK_CPU_INPUT(bsizes_tensor);

  return multimatmul_cuda_forward(weights_tensor, biases_tensor, inputs_tensor, bsizes_tensor, aux_index);
}

torch::Tensor multimatmul_backward_inputs(
    const torch::Tensor& grads_tensor,
    const torch::Tensor& weights_tensor,
    const torch::Tensor& bsizes_tensor,
    int aux_index) {
  CHECK_CUDA_INPUT(grads_tensor);
  CHECK_CUDA_INPUT(weights_tensor);
  CHECK_CPU_INPUT(bsizes_tensor);

  return multimatmul_cuda_backward_inputs(grads_tensor, weights_tensor, bsizes_tensor, aux_index);
}

torch::Tensor multimatmul_backward_weights(
    const torch::Tensor& grads_tensor,
    const torch::Tensor& inputs_tensor,
    const torch::Tensor& bsizes_tensor,
    int aux_index) {
  CHECK_CUDA_INPUT(grads_tensor);
  CHECK_CUDA_INPUT(inputs_tensor);
  CHECK_CPU_INPUT(bsizes_tensor);

  return multimatmul_cuda_backward_weights(grads_tensor, inputs_tensor, bsizes_tensor, aux_index);
}

torch::Tensor multimatmul_backward_biases(
    const torch::Tensor& grads_tensor,
    const torch::Tensor& bsizes_tensor,
    int aux_index) {
  CHECK_CUDA_INPUT(grads_tensor);
  CHECK_CPU_INPUT(bsizes_tensor);

  return multimatmul_cuda_backward_biases(grads_tensor, bsizes_tensor, aux_index);
}

void init_stream_pool(int64_t num_streams) {
  StreamPool::init(num_streams);
}

void destroy_stream_pool() {
  StreamPool::destroy();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // CUDA management
  m.def("init_stream_pool", &init_stream_pool, "Initialize stream pool");
  m.def("destroy_stream_pool", &destroy_stream_pool, "Destroy stream pool");
  m.def("init_magma", &init_magma, "Initialize MAGMA");
  
  m.def("init_multimatmul_aux_data", &init_multimatmul_aux_data, "Create and initialize auxiliary data for multimatmul.");
  m.def("deinit_multimatmul_aux_data", &deinit_multimatmul_aux_data, "Deallocate auxiliary data for multimatmul.");

  // NeRF library functions
  m.def("global_to_local", &global_to_local, "Map global to local coordinates");
  m.def("multimatmul_forward", &multimatmul_forward, "Perform multiple varible-sized matrix multiplication in parallel.");
  m.def("multimatmul_backward_inputs", &multimatmul_backward_inputs, "");
  m.def("multimatmul_backward_weights", &multimatmul_backward_weights, "");
  m.def("multimatmul_backward_biases", &multimatmul_backward_biases, "");
}
