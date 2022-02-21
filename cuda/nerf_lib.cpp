#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void global_to_local_cuda(
    const torch::Tensor& points_tensor,
    const torch::Tensor& mid_points_tensor,
    const torch::Tensor& batch_size_per_network_tensor);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void global_to_local(
    const torch::Tensor& points_tensor,
    const torch::Tensor& mid_points_tensor,
    const torch::Tensor& batch_size_per_network_tensor) {
  CHECK_INPUT(points_tensor);
  CHECK_INPUT(mid_points_tensor);
  CHECK_INPUT(batch_size_per_network_tensor);

  global_to_local_cuda(points_tensor, mid_points_tensor, batch_size_per_network_tensor);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("global_to_local", &global_to_local, "Map global to local coordinates");
}
