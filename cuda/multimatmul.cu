#include "multimatmul.cuh"

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
    // Each group has at most num_nets networks
    magma_int_t int_buf_size = num_groups * (num_nets + 1);
    magma_int_t ptr_buf_size = num_groups * num_nets * sizeof(float*);

    magma_imalloc_cpu(&aux_data.nets_per_group, num_groups);

    // A, B, C buffers
    magma_malloc_cpu((void**)&aux_data.h_A_array, ptr_buf_size);
    magma_malloc_cpu((void**)&aux_data.h_B_array, ptr_buf_size);
    magma_malloc_cpu((void**)&aux_data.h_C_array, ptr_buf_size);
    magma_malloc((void**)&aux_data.d_A_array, ptr_buf_size);
    magma_malloc((void**)&aux_data.d_B_array, ptr_buf_size);
    magma_malloc((void**)&aux_data.d_C_array, ptr_buf_size);

    // M, N, K buffers
    magma_imalloc_cpu(&aux_data.h_M, int_buf_size);
    magma_imalloc_cpu(&aux_data.h_N, int_buf_size);
    magma_imalloc_cpu(&aux_data.h_K, int_buf_size);
    magma_imalloc(&aux_data.d_M, int_buf_size);
    magma_imalloc(&aux_data.d_N, int_buf_size);
    magma_imalloc(&aux_data.d_K, int_buf_size);

    magma_imalloc_cpu(&aux_data.max_m, num_groups);
    magma_imalloc_cpu(&aux_data.max_n, num_groups);
    magma_imalloc_cpu(&aux_data.max_k, num_groups);

    // Batch sizes, offsets
    magma_imalloc_cpu(&aux_data.h_bsizes, num_nets);
    magma_imalloc_cpu(&aux_data.h_offsets, num_nets);
    magma_imalloc(&aux_data.d_bsizes, num_nets);
    magma_imalloc(&aux_data.d_offsets, num_nets);

    // Initalize values constant for all operations, and copy to device
    // For each operation, M / K is always # of out / in channels
    for (int i = 0; i < int_buf_size; ++i) {
        aux_data.h_M[i] = out_channels;
        aux_data.h_K[i] = in_channels;
    }

    for (int i = 0; i < num_groups; ++i) {
        aux_data.max_m[i] = out_channels;
        aux_data.max_k[i] = in_channels;
    }
    
    magma_setvector(int_buf_size, sizeof(magma_int_t), aux_data.h_M, 1, aux_data.d_M, 1, magma_queue);
    magma_setvector(int_buf_size, sizeof(magma_int_t), aux_data.h_K, 1, aux_data.d_K, 1, magma_queue);
    
    // Store initalized aux data
    aux_data_map[aux_data_counter++] = aux_data;
    return aux_data_counter - 1;
}

void deinit_multimatmul_aux_data() {
    for (int i = 0; i < aux_data_counter; ++i) {
        multimatmul_aux_data aux_data = aux_data_map[i];
    
        magma_free_cpu(aux_data.nets_per_group);
    
        magma_free_cpu(aux_data.h_A_array);
        magma_free_cpu(aux_data.h_B_array);
        magma_free_cpu(aux_data.h_C_array);
        magma_free(aux_data.d_A_array);
        magma_free(aux_data.d_B_array);
        magma_free(aux_data.d_C_array);
    
        magma_free_cpu(aux_data.h_M);
        magma_free_cpu(aux_data.h_N);
        magma_free_cpu(aux_data.h_K);
        magma_free(aux_data.d_M);
        magma_free(aux_data.d_N);
        magma_free(aux_data.d_K);
    
        magma_free_cpu(aux_data.max_m);
        magma_free_cpu(aux_data.max_n);
        magma_free_cpu(aux_data.max_k);

        magma_free_cpu(aux_data.h_bsizes);
        magma_free_cpu(aux_data.h_offsets);
        magma_free(aux_data.d_bsizes);
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

torch::Tensor multimatmul_cuda(
    const torch::Tensor& weights_tensor,
    const torch::Tensor& biases_tensor,
    const torch::Tensor& inputs_tensor,
    const torch::Tensor& bsizes_tensor,
    std::vector<int> group_limits,
    int aux_index
) {
    multimatmul_aux_data aux_data = aux_data_map[aux_index];
    
    int64_t num_groups = group_limits.size() + 1;
    int64_t num_nets = bsizes_tensor.size(0);
    int64_t num_pts = inputs_tensor.size(0);
    int64_t out_channels = weights_tensor.size(1);
    int64_t in_channels = weights_tensor.size(2);

    magma_int_t int_buf_size = num_groups * (num_nets + 1);
    magma_int_t ptr_buf_size = num_groups * num_nets;
    
    for (int i = 0; i < num_groups; i++) {
       aux_data.nets_per_group[i] = 0;
       aux_data.max_n[i] = 0;
    }

    torch::Tensor results_tensor = torch::empty({num_pts, out_channels}, inputs_tensor.options());

    auto weights_tensor_a = weights_tensor.accessor<float, 3>();
    auto inputs_tensor_a = inputs_tensor.accessor<float, 2>();
    auto results_tensor_a = results_tensor.accessor<float, 2>();
    auto bsizes_tensor_a = bsizes_tensor.accessor<int64_t, 1>();

    int64_t pt_idx = 0;
    for (int net_idx = 0; net_idx < num_nets; ++net_idx) {
        int64_t bsize = bsizes_tensor_a[net_idx];
        
        aux_data.h_bsizes[net_idx] = bsize;
        aux_data.h_offsets[net_idx] = pt_idx;
        
        if (bsize > 0) {
            int group = num_groups - 1;
            for (int i = 0; i < num_groups - 1; ++i) {
                if (bsize >= group_limits[i]) {
                    group = i;
                    break;
                }
            }

            int int_dest_idx = group * (num_nets + 1) + aux_data.nets_per_group[group];
            int ptr_dest_idx = group * num_nets + aux_data.nets_per_group[group];
        
            aux_data.h_N[int_dest_idx] = bsize;
            aux_data.h_A_array[ptr_dest_idx] = weights_tensor_a[net_idx].data();
            aux_data.h_B_array[ptr_dest_idx] = inputs_tensor_a[pt_idx].data();
            aux_data.h_C_array[ptr_dest_idx] = results_tensor_a[pt_idx].data();
            
            aux_data.nets_per_group[group] += 1;
            if (bsize > aux_data.max_n[group])
                aux_data.max_n[group] = bsize;
        }
        
        pt_idx += bsize;
    }

    magma_setvector(int_buf_size, sizeof(magma_int_t), aux_data.h_N, 1, aux_data.d_N, 1, magma_queue);
    magma_setvector(ptr_buf_size, sizeof(float*), aux_data.h_A_array, 1, aux_data.d_A_array, 1, magma_queue);
    magma_setvector(ptr_buf_size, sizeof(float*), aux_data.h_B_array, 1, aux_data.d_B_array, 1, magma_queue);
    magma_setvector(ptr_buf_size, sizeof(float*), aux_data.h_C_array, 1, aux_data.d_C_array, 1, magma_queue);
    magma_setvector(num_nets, sizeof(magma_int_t), aux_data.h_bsizes, 1, aux_data.d_bsizes, 1, magma_queue);
    magma_setvector(num_nets, sizeof(magma_int_t), aux_data.h_offsets, 1, aux_data.d_offsets, 1, magma_queue);
    
    int max_threads = 1024;
    int y_threads_per_block = max_threads / out_channels;
    int blocks_per_net = 4;
    
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();    
    expand_bias_kernel<<<dim3(num_nets, blocks_per_net), dim3(out_channels, y_threads_per_block), 0, stream>>>(
        results_tensor.packed_accessor64<float, 2>(),
        biases_tensor.packed_accessor64<float, 3>(),
        aux_data.d_bsizes,
        aux_data.d_offsets
    );
    
    float alpha = MAGMA_S_MAKE(1.0, 0.0);
    float beta = MAGMA_S_MAKE(1.0, 0.0);
    
    int queue_idx = 0;
    for (int i = 0; i < num_groups; ++i) {
        if (aux_data.nets_per_group[i] > 0) {
            int ptr_buf_offset = i * num_nets;
            int int_buf_offset = i * (num_nets + 1);

            magmablas_sgemm_vbatched_max_nocheck(
                MagmaTrans, MagmaNoTrans,
                &aux_data.d_M[int_buf_offset], &aux_data.d_N[int_buf_offset], &aux_data.d_K[int_buf_offset],
                alpha, &aux_data.d_A_array[ptr_buf_offset], &aux_data.d_K[int_buf_offset],
                       &aux_data.d_B_array[ptr_buf_offset], &aux_data.d_K[int_buf_offset],
                beta,  &aux_data.d_C_array[ptr_buf_offset], &aux_data.d_M[int_buf_offset],
                aux_data.nets_per_group[i], aux_data.max_m[i], aux_data.max_n[i], aux_data.max_k[i],
                magma_streams[queue_idx]
            );
            
            queue_idx = (queue_idx + 1) % StreamPool::num_streams;
        }
    }
    
    return results_tensor;
}
