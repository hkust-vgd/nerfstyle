#ifndef __MULTIMATMUL__
#define __MULTIMATMUL__

#include <torch/extension.h>
#include <ATen/cuda/CUDABlas.h>
#include "streams.cuh"
#include "magma_v2.h"

enum dim {
    in_dim,
    out_dim,
    bsize_dim
};

// Data struct for holding parameters to the MAGMA SGEMM operation.
// One for each MultiLinear layer.
struct multimatmul_aux_data {
    int num_groups, num_nets;
    int out_channels, in_channels;
    std::vector<int> group_limits;

    // No. of networks per grouped operation
    magma_int_t* nets_per_group;

    std::vector<magma_int_t*> h_out, h_in, h_bsize;
    std::vector<magma_int_t*> d_out, d_in, d_bsize;
    magma_int_t *max_out, *max_in, *max_bsize;

    std::vector<float**> h_A_array, h_B_array, h_C_array;
    std::vector<float**> d_A_array, d_B_array, d_C_array;

    float **h_x_array, **h_y_array;
    float **d_x_array, **d_y_array;
    magma_int_t *h_inc, *d_inc;

    // Batch size per network, offsets
    // Used in the kernel for copying bias vectors to respective locations before SGEMM operation
    magma_int_t *h_flat_bsizes, *h_offsets;
    magma_int_t *d_flat_bsizes, *d_offsets;
};

void init_magma();
int init_multimatmul_aux_data(int64_t num_nets, int64_t out_features, int64_t in_features, std::vector<int> group_limits);
void deinit_multimatmul_aux_data();

torch::Tensor multimatmul_cuda_forward(
    const torch::Tensor& weights_tensor,
    const torch::Tensor& biases_tensor,
    const torch::Tensor& inputs_tensor,
    const torch::Tensor& bsizes_tensor,
    int aux_index
);

torch::Tensor multimatmul_cuda_backward_inputs(
    const torch::Tensor& grads_tensor,
    const torch::Tensor& weights_tensor,
    const torch::Tensor& bsizes_tensor,
    int aux_index
);

torch::Tensor multimatmul_cuda_backward_weights(
    const torch::Tensor& grads_tensor,
    const torch::Tensor& inputs_tensor,
    const torch::Tensor& bsizes_tensor,
    int aux_index
);

torch::Tensor multimatmul_cuda_backward_biases(
    const torch::Tensor& grads_tensor,
    const torch::Tensor& bsizes_tensor,
    int aux_index
);

#endif
