#include "multimatmul.cuh"
#include <cstdio>

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
    int64_t num_networks,
    int64_t out_features,
    int64_t in_features,
    std::vector<int> group_limits
) {
    multimatmul_aux_data aux_data;

    int num_groups = group_limits.size() + 1;
    // Each group has at most num_networks networks
    int64_t int_buf_size = num_groups * (num_networks + 1); // M, N, K, LDDs
    int64_t ptr_buf_size = num_groups * num_networks * sizeof(float*); // A, B, C

    magma_imalloc_cpu(&aux_data.nets_per_group, num_groups);

    // A, B, C buffers
    magma_malloc_cpu((void**)&aux_data.h_A_array, ptr_buf_size);
    magma_malloc_cpu((void**)&aux_data.h_B_array, ptr_buf_size);
    magma_malloc_cpu((void**)&aux_data.h_C_array, ptr_buf_size);
    magma_malloc((void**)&aux_data.d_A_array, ptr_buf_size);
    magma_malloc((void**)&aux_data.d_B_array, ptr_buf_size);
    magma_malloc((void**)&aux_data.d_C_array, ptr_buf_size);

    // LDD buffers
    magma_imalloc_cpu(&aux_data.h_ldda, int_buf_size);
    magma_imalloc_cpu(&aux_data.h_lddb, int_buf_size);
    magma_imalloc_cpu(&aux_data.h_lddc, int_buf_size);
    magma_imalloc(&aux_data.d_ldda, int_buf_size);
    magma_imalloc(&aux_data.d_lddb, int_buf_size);
    magma_imalloc(&aux_data.d_lddc, int_buf_size);

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

    // Initalize values constant for all operations, and copy to device
    // For each operation, M / K is always # of out / in channels
    for (int i = 0; i < int_buf_size; ++i) {
        aux_data.h_M[i] = out_features;
        aux_data.h_K[i] = in_features;
        aux_data.h_ldda[i] = out_features; // A has M rows
        aux_data.h_ldda[i] = in_features; // B has K rows
        aux_data.h_ldda[i] = out_features; // C has M rows
    }

    for (int i = 0; i < num_groups; ++i) {
        aux_data.max_m[i] = out_features;
        aux_data.max_k[i] = in_features;
    }
    
    magma_setvector(int_buf_size, sizeof(magma_int_t), aux_data.h_M, 1, aux_data.d_M, 1, magma_queue);
    magma_setvector(int_buf_size, sizeof(magma_int_t), aux_data.h_K, 1, aux_data.d_K, 1, magma_queue);
    magma_setvector(int_buf_size, sizeof(magma_int_t), aux_data.h_ldda, 1, aux_data.d_ldda, 1, magma_queue);
    magma_setvector(int_buf_size, sizeof(magma_int_t), aux_data.h_lddb, 1, aux_data.d_lddb, 1, magma_queue);
    magma_setvector(int_buf_size, sizeof(magma_int_t), aux_data.h_lddc, 1, aux_data.d_lddc, 1, magma_queue);
    
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
    
        magma_free_cpu(aux_data.h_ldda);
        magma_free_cpu(aux_data.h_lddb);
        magma_free_cpu(aux_data.h_lddc);
        magma_free(aux_data.d_ldda);
        magma_free(aux_data.d_lddb);
        magma_free(aux_data.d_lddc);
    
        magma_free_cpu(aux_data.h_M);
        magma_free_cpu(aux_data.h_N);
        magma_free_cpu(aux_data.h_K);
        magma_free(aux_data.d_M);
        magma_free(aux_data.d_N);
        magma_free(aux_data.d_K);
    
        magma_free_cpu(aux_data.max_m);
        magma_free_cpu(aux_data.max_n);
        magma_free_cpu(aux_data.max_k);
    
        aux_data_map.erase(i);
    }
}

void multimatmul_cuda(
    const torch::Tensor& weights_tensor,
    const torch::Tensor& biases_tensor,
    const torch::Tensor& inputs_tensor,
    const torch::Tensor& bsizes_tensor,
    std::vector<int> group_limits,
    int aux_index
) {
    multimatmul_aux_data aux_data = aux_data_map[aux_index];

    int num_groups = group_limits.size() + 1;
    int num_networks = bsizes_tensor.size(0);

    for (int i = 0; i < num_groups; ++i) {
        aux_data.nets_per_group[i] = 0;
        aux_data.max_n[i] = 0;
    }

    auto weights_tensor_a = weights_tensor.accessor<float, 3>();
    auto inputs_tensor_a = inputs_tensor.accessor<float, 2>();
    auto bsizes_tensor_a = bsizes_tensor.accessor<int64_t, 1>();

    for (int i = 0; i < num_networks; ++i) {
        int64_t bsize = bsizes_tensor_a[i];

        if (bsize > 0) {
            int group = num_groups - 1;
            for (int j = 0; j < num_groups - 1; ++j) {
                if (bsize >= group_limits[j]) {
                    group = j;
                }
            }
            
            int dest_idx = group * (num_networks + 1) + aux_data.nets_per_group[group];
            aux_data.h_N[dest_idx] = bsize;
            aux_data.h_A_array[dest_idx] = weights_tensor_a[i].data();

            aux_data.nets_per_group[group]++;
            if (bsize > aux_data.max_n[group])
                aux_data.max_n[group] = bsize;
        }
    }
}
