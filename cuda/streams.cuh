#ifndef __STREAMS__
#define __STREAMS__

#include <cuda_runtime.h>

class StreamPool {
public:
    static void init(int64_t num_streams);
    static void destroy();
    static cudaStream_t get_next_stream();

    static cudaStream_t* streamArray;
    static int64_t num_streams;
protected:
    static int64_t counter;
};

#endif
