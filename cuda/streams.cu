#include "streams.cuh"
#include <iostream>

cudaStream_t *StreamPool::streamArray = nullptr;
int64_t StreamPool::counter = 0;
int64_t StreamPool::num_streams = 0;

void StreamPool::init(int64_t num_streams) {
    StreamPool::streamArray = (cudaStream_t*)malloc(num_streams * sizeof(cudaStream_t*));
    StreamPool::num_streams = num_streams;
    StreamPool::counter = 0;

    for (int i = 0; i < num_streams; i++) {
        cudaError_t cudaErr = cudaStreamCreate(&StreamPool::streamArray[i]);
        if (cudaErr != cudaSuccess)
            std::cerr << "Cannot create stream " << i << std::endl;
    }
}

void StreamPool::destroy() {
    free(streamArray);
}

cudaStream_t StreamPool::get_next_stream() {
    cudaStream_t cur_stream = streamArray[counter];
    counter = (counter + 1) % num_streams;
    return cur_stream;
}
