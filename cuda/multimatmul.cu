#include "multimatmul.cuh"
#include <iostream>

magma_queue_t magma_queue = NULL;
magma_queue_t *magma_streams = NULL;

std::map<int, multimatmul_aux_data> aux_data_map;
int aux_data_counter = 0;

void init_magma() {
    magma_init();
    magma_int_t dev = 0;

    cudaStream_t cuda_stream = at::cuda::getDefaultCUDAStream();
    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    magma_queue_create_from_cuda(dev, cuda_stream, cublas_handle, NULL, &magma_queue);

    magma_streams = new magma_queue_t[StreamPool::num_streams];
    for (int i = 0; i < StreamPool::num_streams; ++i) {
        magma_queue_create_from_cuda(dev, StreamPool::streamArray[i], cublas_handle, NULL, &magma_streams[i]);
    }
}

// Each of the N networks are put in a group, depending on its batch size.
// Networks in the same group will be evaluated in the same operation.

// group_limits: list of descending ints for determining groups.
// e.g. group_limits {A, B} create 3 groups:
// - All networks with at least A points to evaluate -> [A, inf)
// - All networks with at least B points and no more than A points to evaluate -> [B, A)
// - All networks with no more than B points to evaluate -> [1, B)
int init_multimatmul_aux_data(
    int64_t num_nets,
    int64_t out_channels,
    int64_t in_channels,
    std::vector<int> group_limits
) {
    multimatmul_aux_data aux_data;

    int num_groups = group_limits.size() + 1;
    aux_data.num_groups = num_groups;
    aux_data.num_nets = num_nets;
    aux_data.out_channels = out_channels;
    aux_data.in_channels = in_channels;
    aux_data.group_limits = group_limits;

    magma_imalloc_cpu(&aux_data.nets_per_group, num_groups);

    for (int i = 0; i < num_groups; ++i) {
        // A, B, C buffers
        aux_data.h_A_array.push_back(nullptr);
        aux_data.h_B_array.push_back(nullptr);
        aux_data.h_C_array.push_back(nullptr);
        aux_data.d_A_array.push_back(nullptr);
        aux_data.d_B_array.push_back(nullptr);
        aux_data.d_C_array.push_back(nullptr);

        magma_malloc_cpu((void**)&aux_data.h_A_array[i], num_nets * sizeof(float*));
        magma_malloc_cpu((void**)&aux_data.h_B_array[i], num_nets * sizeof(float*));
        magma_malloc_cpu((void**)&aux_data.h_C_array[i], num_nets * sizeof(float*));
        magma_malloc((void**)&aux_data.d_A_array[i], num_nets * sizeof(float*));
        magma_malloc((void**)&aux_data.d_B_array[i], num_nets * sizeof(float*));
        magma_malloc((void**)&aux_data.d_C_array[i], num_nets * sizeof(float*));

        // M, N, K buffers
        aux_data.h_out.push_back(nullptr);
        aux_data.h_in.push_back(nullptr);
        aux_data.h_bsize.push_back(nullptr);
        aux_data.d_out.push_back(nullptr);
        aux_data.d_in.push_back(nullptr);
        aux_data.d_bsize.push_back(nullptr);

        magma_imalloc_cpu(&aux_data.h_out[i], num_nets + 1);
        magma_imalloc_cpu(&aux_data.h_in[i], num_nets + 1);
        magma_imalloc_cpu(&aux_data.h_bsize[i], num_nets + 1);
        magma_imalloc(&aux_data.d_out[i], num_nets + 1);
        magma_imalloc(&aux_data.d_in[i], num_nets + 1);
        magma_imalloc(&aux_data.d_bsize[i], num_nets + 1);

        // Initialize all M to out_channels and all K to in_channels
        for (int j = 0; j < (num_nets + 1); ++j) {
            aux_data.h_out[i][j] = out_channels;
            aux_data.h_in[i][j] = in_channels;
        }
    
        magma_setvector(num_nets + 1, sizeof(magma_int_t), aux_data.h_out[i], 1, aux_data.d_out[i], 1, magma_queue);
        magma_setvector(num_nets + 1, sizeof(magma_int_t), aux_data.h_in[i], 1, aux_data.d_in[i], 1, magma_queue);
    }

    magma_malloc_cpu((void**)&aux_data.h_x_array, num_nets * sizeof(float*));
    magma_malloc_cpu((void**)&aux_data.h_y_array, num_nets * sizeof(float*));
    magma_malloc((void**)&aux_data.d_x_array, num_nets * sizeof(float*));
    magma_malloc((void**)&aux_data.d_y_array, num_nets * sizeof(float*));

    magma_imalloc_cpu(&aux_data.h_inc, num_nets + 1);
    magma_imalloc(&aux_data.d_inc, num_nets + 1);

    magma_imalloc_cpu(&aux_data.max_out, num_groups);
    magma_imalloc_cpu(&aux_data.max_in, num_groups);
    magma_imalloc_cpu(&aux_data.max_bsize, num_groups);
    
    for (int i = 0; i < num_groups; ++i) {
        aux_data.max_out[i] = out_channels;
        aux_data.max_in[i] = in_channels;
    }

    for (int i = 0; i < num_nets; ++i) {
        aux_data.h_inc[i] = 1;
    }

    magma_setvector(num_nets + 1, sizeof(magma_int_t), aux_data.h_inc, 1, aux_data.d_inc, 1, magma_queue);

    // Batch sizes, offsets
    magma_imalloc_cpu(&aux_data.h_flat_bsizes, num_nets);
    magma_imalloc_cpu(&aux_data.h_offsets, num_nets);
    magma_imalloc(&aux_data.d_flat_bsizes, num_nets);
    magma_imalloc(&aux_data.d_offsets, num_nets);
    
    // Store initalized aux data
    aux_data_map[aux_data_counter++] = aux_data;
    return aux_data_counter - 1;
}

void deinit_multimatmul_aux_data() {
    for (int i = 0; i < aux_data_counter; ++i) {
        multimatmul_aux_data aux_data = aux_data_map[i];
    
        magma_free_cpu(aux_data.nets_per_group);
        
        for (int j = 0; j < aux_data.num_groups; ++j) {
            magma_free_cpu(aux_data.h_A_array[j]);
            magma_free_cpu(aux_data.h_B_array[j]);
            magma_free_cpu(aux_data.h_C_array[j]);
            magma_free(aux_data.d_A_array[j]);
            magma_free(aux_data.d_B_array[j]);
            magma_free(aux_data.d_C_array[j]);
    
            magma_free_cpu(aux_data.h_out[j]);
            magma_free_cpu(aux_data.h_in[j]);
            magma_free_cpu(aux_data.h_bsize[j]);
            magma_free(aux_data.d_out[j]);
            magma_free(aux_data.d_in[j]);
            magma_free(aux_data.d_bsize[j]);
        }

        magma_free_cpu(aux_data.h_x_array);
        magma_free_cpu(aux_data.h_y_array);
        magma_free_cpu(aux_data.h_inc);
        magma_free(aux_data.d_x_array);
        magma_free(aux_data.d_y_array);
        magma_free(aux_data.d_inc);
    
        magma_free_cpu(aux_data.max_out);
        magma_free_cpu(aux_data.max_in);
        magma_free_cpu(aux_data.max_bsize);

        magma_free_cpu(aux_data.h_flat_bsizes);
        magma_free_cpu(aux_data.h_offsets);
        magma_free(aux_data.d_flat_bsizes);
        magma_free(aux_data.d_offsets);
    
        aux_data_map.erase(i);
    }
}

__global__ void expand_bias_kernel(
    torch::PackedTensorAccessor64<float, 2> results,
    const torch::PackedTensorAccessor64<float, 3> biases,
    const int* bsizes,
    const int* offsets
) {
    int net_idx = blockIdx.x;
    int pt_idx = blockDim.y * blockIdx.y + threadIdx.y;
    int elem_idx = threadIdx.x;
    int pt_offset = offsets[net_idx];
    
    while (pt_idx < bsizes[net_idx]) {
        results[pt_offset + pt_idx][elem_idx] = biases[net_idx][0][elem_idx];
        pt_idx += gridDim.y * blockDim.y;
    }
}

void _set_dimensions(std::vector<magma_int_t*>& ref, magma_int_t** max_ref, const multimatmul_aux_data& aux_data, dim d) {
    switch(d) {
        case in_dim: ref = aux_data.d_in; *max_ref = aux_data.max_in; break;
        case out_dim: ref = aux_data.d_out; *max_ref = aux_data.max_out; break;
        case bsize_dim: ref = aux_data.d_bsize; *max_ref = aux_data.max_bsize; break;
    }
}

int _get_group_id(int bsize, const multimatmul_aux_data& aux_data) {
    int group = aux_data.num_groups - 1;
    for (int i = 0; i < aux_data.num_groups - 1; ++i) {
        if (bsize >= aux_data.group_limits[i]) {
            group = i;
            break;
        }
    }
    return group;
}

void _multimatmul_cuda(
    const torch::Tensor& results_tensor,
    const torch::Tensor& biases_tensor,
    const torch::Tensor& bsizes_tensor,
    dim m_dim,
    dim n_dim,
    dim k_dim,
    bool transpose_A,
    bool transpose_B,
    bool use_bias,
    int aux_index
) {
    multimatmul_aux_data aux_data = aux_data_map[aux_index];
    int num_nets = aux_data.num_nets;
    int out_channels = aux_data.out_channels;
    auto bsizes_tensor_a = bsizes_tensor.accessor<int64_t, 1>();

    std::vector<magma_int_t*> d_M, d_N, d_K;
    magma_int_t* max_m = nullptr;
    magma_int_t* max_n = nullptr;
    magma_int_t* max_k = nullptr;
    _set_dimensions(d_M, &max_m, aux_data, m_dim);
    _set_dimensions(d_N, &max_n, aux_data, n_dim);
    _set_dimensions(d_K, &max_k, aux_data, k_dim);

    magma_trans_t transA = transpose_A ? MagmaTrans : MagmaNoTrans;
    magma_trans_t transB = transpose_B ? MagmaTrans : MagmaNoTrans;
    std::vector<magma_int_t*> d_lddA = transpose_A ? d_K : d_M;
    std::vector<magma_int_t*> d_lddB = transpose_B ? d_N : d_K;
    
    for (int i = 0; i < aux_data.num_groups; ++i) {
        magma_setvector(num_nets + 1, sizeof(magma_int_t), aux_data.h_bsize[i], 1, aux_data.d_bsize[i], 1, magma_queue);
        magma_setvector(num_nets, sizeof(float*), aux_data.h_A_array[i], 1, aux_data.d_A_array[i], 1, magma_queue);
        magma_setvector(num_nets, sizeof(float*), aux_data.h_B_array[i], 1, aux_data.d_B_array[i], 1, magma_queue);
        magma_setvector(num_nets, sizeof(float*), aux_data.h_C_array[i], 1, aux_data.d_C_array[i], 1, magma_queue);
    }
    
    if (use_bias) {
        int bsize;
        for (int net_idx = 0, pt_idx = 0; net_idx < num_nets; ++net_idx, pt_idx += bsize) {
            bsize = bsizes_tensor_a[net_idx];
            aux_data.h_flat_bsizes[net_idx] = bsize;
            aux_data.h_offsets[net_idx] = pt_idx;
        }

        magma_setvector(num_nets, sizeof(magma_int_t), aux_data.h_flat_bsizes, 1, aux_data.d_flat_bsizes, 1, magma_queue);
        magma_setvector(num_nets, sizeof(magma_int_t), aux_data.h_offsets, 1, aux_data.d_offsets, 1, magma_queue);

        int max_threads = 1024;
        int y_threads_per_block = max_threads / out_channels;
        int blocks_per_net = 4;

        cudaStream_t stream = at::cuda::getDefaultCUDAStream();    
        expand_bias_kernel<<<dim3(num_nets, blocks_per_net), dim3(out_channels, y_threads_per_block), 0, stream>>>(
            results_tensor.packed_accessor64<float, 2>(),
            biases_tensor.packed_accessor64<float, 3>(),
            aux_data.d_flat_bsizes,
            aux_data.d_offsets
        );
    }
    
    float alpha = MAGMA_S_MAKE(1.0, 0.0);
    float beta = MAGMA_S_MAKE(use_bias ? 1.0 : 0.0, 0.0);
    
    int queue_idx = 0;
    for (int i = 0; i < aux_data.num_groups; ++i) {
        if (aux_data.nets_per_group[i] > 0) {
            magmablas_sgemm_vbatched_max_nocheck(
                transA, transB, &d_M[i][0], &d_N[i][0], &d_K[i][0],
                alpha, &aux_data.d_A_array[i][0], &d_lddA[i][0],
                       &aux_data.d_B_array[i][0], &d_lddB[i][0],
                beta,  &aux_data.d_C_array[i][0], &d_M[i][0],
                aux_data.nets_per_group[i], max_m[i], max_n[i], max_k[i],
                magma_streams[queue_idx]
            );
            
            queue_idx = (queue_idx + 1) % StreamPool::num_streams;
        }
    }
}

void _multimatreduce_cuda(
    dim m_dim,
    dim n_dim,
    int aux_index
) {
    multimatmul_aux_data aux_data = aux_data_map[aux_index];
    int num_nets = aux_data.num_nets;

    std::vector<magma_int_t*> d_M, d_N;
    magma_int_t* tmp = nullptr;
    _set_dimensions(d_M, &tmp, aux_data, m_dim);
    _set_dimensions(d_N, &tmp, aux_data, n_dim);
    
    magma_setvector(num_nets + 1, sizeof(magma_int_t), aux_data.h_bsize[0], 1, aux_data.d_bsize[0], 1, magma_queue);
    magma_setvector(num_nets, sizeof(float*), aux_data.h_A_array[0], 1, aux_data.d_A_array[0], 1, magma_queue);
    magma_setvector(num_nets, sizeof(float*), aux_data.h_x_array, 1, aux_data.d_x_array, 1, magma_queue);
    magma_setvector(num_nets, sizeof(float*), aux_data.h_y_array, 1, aux_data.d_y_array, 1, magma_queue);
    
    float alpha = MAGMA_S_MAKE(1.0, 0.0);
    float beta = MAGMA_S_MAKE(0.0, 0.0);

    magmablas_sgemv_vbatched(
        MagmaNoTrans, &d_M[0][0], &d_N[0][0],
        alpha, &aux_data.d_A_array[0][0], &d_M[0][0],
               &aux_data.d_x_array[0], &aux_data.d_inc[0],
        beta,  &aux_data.d_y_array[0], &aux_data.d_inc[0],
        aux_data.num_nets, magma_queue
    );
}

torch::Tensor multimatmul_cuda_forward(
    const torch::Tensor& weights_tensor,
    const torch::Tensor& biases_tensor,
    const torch::Tensor& inputs_tensor,
    const torch::Tensor& bsizes_tensor,
    int aux_index
) {
    multimatmul_aux_data aux_data = aux_data_map[aux_index];
    int num_pts = inputs_tensor.size(0);

    torch::Tensor results_tensor = torch::zeros({num_pts, aux_data.out_channels}, inputs_tensor.options());

    auto weights_tensor_a = weights_tensor.accessor<float, 3>();
    auto inputs_tensor_a = inputs_tensor.accessor<float, 2>();
    auto results_tensor_a = results_tensor.accessor<float, 2>();
    auto bsizes_tensor_a = bsizes_tensor.accessor<int64_t, 1>();
    
    for (int i = 0; i < aux_data.num_groups; ++i) {
        aux_data.nets_per_group[i] = 0;
        aux_data.max_bsize[i] = 0;
    }
    
    int bsize;
    for (int net_idx = 0, pt_idx = 0; net_idx < aux_data.num_nets; ++net_idx, pt_idx += bsize) {
        bsize = bsizes_tensor_a[net_idx];        
        if (bsize > 0) {
            int group = _get_group_id(bsize, aux_data);
            int group_net_idx = aux_data.nets_per_group[group];
            aux_data.h_bsize[group][group_net_idx] = bsize;
            aux_data.h_A_array[group][group_net_idx] = weights_tensor_a[net_idx].data();
            aux_data.h_B_array[group][group_net_idx] = inputs_tensor_a[pt_idx].data();
            aux_data.h_C_array[group][group_net_idx] = results_tensor_a[pt_idx].data();
            
            aux_data.nets_per_group[group] += 1;
            if (bsize > aux_data.max_bsize[group])
                aux_data.max_bsize[group] = bsize;
        }
    }

    // (in, out)^T @ (in, bsize) + (out) -> (out, bsize)
    // M: out, N: bsize, K: in
    _multimatmul_cuda(results_tensor, biases_tensor, bsizes_tensor, out_dim, bsize_dim, in_dim, true, false, true, aux_index);
    return results_tensor;
}

torch::Tensor multimatmul_cuda_backward_inputs(
    const torch::Tensor& grads_tensor,
    const torch::Tensor& weights_tensor,
    const torch::Tensor& bsizes_tensor,
    int aux_index
) {
    multimatmul_aux_data aux_data = aux_data_map[aux_index];
    int num_pts = grads_tensor.size(0);

    torch::Tensor results_tensor = torch::zeros({num_pts, aux_data.in_channels}, grads_tensor.options());

    auto grads_tensor_a = grads_tensor.accessor<float, 2>();
    auto weights_tensor_a = weights_tensor.accessor<float, 3>();
    auto results_tensor_a = results_tensor.accessor<float, 2>();
    auto bsizes_tensor_a = bsizes_tensor.accessor<int64_t, 1>();
    
    for (int i = 0; i < aux_data.num_groups; ++i) {
        aux_data.nets_per_group[i] = 0;
        aux_data.max_bsize[i] = 0;
    }
    
    int bsize;
    for (int net_idx = 0, pt_idx = 0; net_idx < aux_data.num_nets; ++net_idx, pt_idx += bsize) {
        bsize = bsizes_tensor_a[net_idx];        
        if (bsize > 0) {
            int group = _get_group_id(bsize, aux_data);
            int group_net_idx = aux_data.nets_per_group[group];
            aux_data.h_bsize[group][group_net_idx] = bsize;
            aux_data.h_A_array[group][group_net_idx] = weights_tensor_a[net_idx].data();
            aux_data.h_B_array[group][group_net_idx] = grads_tensor_a[pt_idx].data();
            aux_data.h_C_array[group][group_net_idx] = results_tensor_a[pt_idx].data();
            
            aux_data.nets_per_group[group] += 1;
            if (bsize > aux_data.max_bsize[group])
                aux_data.max_bsize[group] = bsize;
        }
    }

    // (in, out) @ (out, bsize) -> (in, bsize)
    // M: in, N: bsize, K: out
    _multimatmul_cuda(results_tensor, results_tensor, bsizes_tensor, in_dim, bsize_dim, out_dim, false, false, false, aux_index);
    return results_tensor;
}

torch::Tensor multimatmul_cuda_backward_weights(
    const torch::Tensor& grads_tensor,
    const torch::Tensor& inputs_tensor,
    const torch::Tensor& bsizes_tensor,
    int aux_index
) {
    multimatmul_aux_data aux_data = aux_data_map[aux_index];

    torch::Tensor results_tensor = torch::zeros({aux_data.num_nets, aux_data.out_channels, aux_data.in_channels}, grads_tensor.options());

    auto grads_tensor_a = grads_tensor.accessor<float, 2>();
    auto inputs_tensor_a = inputs_tensor.accessor<float, 2>();
    auto results_tensor_a = results_tensor.accessor<float, 3>();
    auto bsizes_tensor_a = bsizes_tensor.accessor<int64_t, 1>();
    
    for (int i = 0; i < aux_data.num_groups; ++i) {
        aux_data.nets_per_group[i] = 0;
        aux_data.max_bsize[i] = 0;
    }
    
    int bsize;
    for (int net_idx = 0, pt_idx = 0; net_idx < aux_data.num_nets; ++net_idx, pt_idx += bsize) {
        bsize = bsizes_tensor_a[net_idx];        
        if (bsize > 0) {
            int group = _get_group_id(bsize, aux_data);
            int group_net_idx = aux_data.nets_per_group[group];
            aux_data.h_bsize[group][group_net_idx] = bsize;
            aux_data.h_A_array[group][group_net_idx] = inputs_tensor_a[pt_idx].data();
            aux_data.h_B_array[group][group_net_idx] = grads_tensor_a[pt_idx].data();
            aux_data.h_C_array[group][group_net_idx] = results_tensor_a[net_idx].data();
            
            aux_data.nets_per_group[group] += 1;
            if (bsize > aux_data.max_bsize[group])
                aux_data.max_bsize[group] = bsize;
        }
    }

    // (in, bsize) @ (out, bsize)^T -> (in, out)
    // M: in, N: out, K: bsize
    _multimatmul_cuda(results_tensor, results_tensor, bsizes_tensor, in_dim, out_dim, bsize_dim, false, true, false, aux_index);
    return results_tensor;
}

torch::Tensor multimatmul_cuda_backward_biases(
    const torch::Tensor& grads_tensor,
    const torch::Tensor& bsizes_tensor,
    int aux_index
) {
    multimatmul_aux_data aux_data = aux_data_map[aux_index];
    int num_pts = grads_tensor.size(0);

    torch::Tensor results_tensor = torch::zeros({aux_data.num_nets, 1, aux_data.out_channels}, grads_tensor.options());
    torch::Tensor ones_tensor = torch::ones({num_pts}, grads_tensor.options());

    auto grads_tensor_a = grads_tensor.accessor<float, 2>();
    auto ones_tensor_a = ones_tensor.accessor<float, 1>();
    auto results_tensor_a = results_tensor.accessor<float, 3>();
    auto bsizes_tensor_a = bsizes_tensor.accessor<int64_t, 1>();

    int bsize;
    for (int net_idx = 0, pt_idx = 0; net_idx < aux_data.num_nets; ++net_idx, pt_idx += bsize) {
        bsize = bsizes_tensor_a[net_idx];        
        if (bsize > 0) {
            aux_data.h_bsize[0][net_idx] = bsize;
            aux_data.h_A_array[0][net_idx] = grads_tensor_a[pt_idx].data();
            aux_data.h_x_array[net_idx] = &ones_tensor_a[pt_idx];
            aux_data.h_y_array[net_idx] = results_tensor_a[net_idx].data();
        }
    }

    // (out, bsize) @ (bsize, 1) -> (out, 1)
    // M: out, N: bsize
    _multimatreduce_cuda(out_dim, bsize_dim, aux_index);
    return results_tensor;
}
