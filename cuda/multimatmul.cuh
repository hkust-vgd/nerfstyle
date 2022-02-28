#ifndef __MULTIMATMUL__
#define __MULTIMATMUL__

#include <torch/extension.h>
#include <ATen/cuda/CUDABlas.h>
#include "streams.cuh"
#include "magma_v2.h"

// Data struct for holding parameters to the MAGMA SGEMM operation.
// One for each MultiLinear layer.
struct multimatmul_aux_data {
    // No. of networks per grouped operation
    magma_int_t* nets_per_group;

    // M: input channels
    // N: no. of points
    // K: output channels
    magma_int_t *h_M, *h_N, *h_K;
    magma_int_t *d_M, *d_N, *d_K;

    // Matrix A: weights (M, K)
    // Matrix B: inputs (K, N)
    // Matrix C: biases (M, N)
    // Note: the row / column dimensions of input tensors are actually flipped, since
    // BLAS / MAGMA use column major and PyTorch use row major
    float **h_A_array = nullptr;
    float **h_B_array = nullptr;
    float **h_C_array = nullptr;
    float **d_A_array = nullptr;
    float **d_B_array = nullptr;
    float **d_C_array = nullptr;

    magma_int_t *h_ldda, *h_lddb, *h_lddc;
    magma_int_t *d_ldda, *d_lddb, *d_lddc;

    // Max values of M, N, K
    magma_int_t *max_m, *max_n, *max_k;

    // Batch size per network, offsets
    // Used in the kernel for copying bias vectors to respective locations before SGEMM operation
    magma_int_t *h_bsizes, *h_offsets;
    magma_int_t *d_bsizes, *d_offsets;
};

void init_magma();
int init_multimatmul_aux_data(int64_t num_nets, int64_t out_features, int64_t in_features, std::vector<int> group_limits);
void deinit_multimatmul_aux_data();

torch::Tensor multimatmul_cuda(
    const torch::Tensor& weights_tensor,
    const torch::Tensor& biases_tensor,
    const torch::Tensor& inputs_tensor,
    const torch::Tensor& bsizes_tensor,
    std::vector<int> group_limits,
    int aux_index
);

#endif
