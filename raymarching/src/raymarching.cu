#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <cstdio>
#include <stdint.h>
#include <stdexcept>
#include <limits>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")


inline constexpr __device__ float SQRT3() { return 1.7320508075688772f; }
inline constexpr __device__ float RSQRT3() { return 0.5773502691896258f; }
inline constexpr __device__ float PI() { return 3.141592653589793f; }
inline constexpr __device__ float RPI() { return 0.3183098861837907f; }


template <typename T>
inline __host__ __device__ T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}

inline __host__ __device__ float signf(const float x) {
    return copysignf(1.0, x);
}

inline __host__ __device__ float clamp(const float x, const float min, const float max) {
    return fminf(max, fmaxf(min, x));
}

inline __host__ __device__ void swapf(float& a, float& b) {
    float c = a; a = b; b = c;
}

inline __device__ int mip_from_pos(const float x, const float y, const float z, const float max_cascade) {
    const float mx = fmaxf(fabsf(x), fmaxf(fabs(y), fabs(z)));
    int exponent;
    frexpf(mx, &exponent); // [0, 0.5) --> -1, [0.5, 1) --> 0, [1, 2) --> 1, [2, 4) --> 2, ...
    return fminf(max_cascade - 1, fmaxf(0, exponent));
}

inline __device__ int mip_from_dt(const float dt, const float H, const float max_cascade) {
    const float mx = dt * H * 0.5;
    int exponent;
    frexpf(mx, &exponent);
    return fminf(max_cascade - 1, fmaxf(0, exponent));
}

inline __host__ __device__ uint32_t __expand_bits(uint32_t v)
{
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

inline __host__ __device__ uint32_t __morton3D(uint32_t x, uint32_t y, uint32_t z)
{
	uint32_t xx = __expand_bits(x);
	uint32_t yy = __expand_bits(y);
	uint32_t zz = __expand_bits(z);
	return xx | (yy << 1) | (zz << 2);
}

inline __host__ __device__ uint32_t __morton3D_invert(uint32_t x)
{
	x = x & 0x49249249;
	x = (x | (x >> 2)) & 0xc30c30c3;
	x = (x | (x >> 4)) & 0x0f00f00f;
	x = (x | (x >> 8)) & 0xff0000ff;
	x = (x | (x >> 16)) & 0x0000ffff;
	return x;
}

// Element-wise operations

inline __host__ __device__ float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 operator*(const float3 &a, const float3 &b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ float fminf(const float3 &v) {
    return fminf(v.x, fminf(v.y, v.z));
}

// Scalar operations

inline __host__ __device__ float3 operator+(const float a, const float3 &b) {
    return make_float3(a + b.x, a + b.y, a + b.z);
}

inline __host__ __device__ float3 operator+(const float3 &a, const float b) {
    return b + a;
}

inline __host__ __device__ float3 operator-(const float a, const float3 &b) {
    return make_float3(a - b.x, a - b.y, a - b.z);
}

inline __host__ __device__ float3 operator-(const float3 &a, const float b) {
    return make_float3(a.x - b, a.y - b, a.z - b);
}

inline __host__ __device__ float3 operator*(const float a, const float3 &b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

inline __host__ __device__ float3 operator*(const float3 &a, const float b) {
    return b * a;
}

inline __host__ __device__ float3 operator/(const float a, const float3 &b) {
    return make_float3(a / b.x, a / b.y, a / b.z);
}

inline __host__ __device__ float3 operator/(const float3 &a, const float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

// Vector math
inline __host__ __device__ float dot(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float norm(const float3 &v) {
    return sqrtf(dot(v, v));
}

// Assignment
template <typename scalar_t>
inline __host__ __device__ void write_float3(const float3 &v, scalar_t * dest) {
    dest[0] = v.x;
    dest[1] = v.y;
    dest[2] = v.z;
}

// Conversion
inline __host__ __device__ const uint3& to_uint3(const float3& v) {
    return make_uint3((uint)v.x, (uint)v.y, (uint)v.z);
}

inline __host__ __device__ const float3& to_float3(const uint3& v) {
    return make_float3((float)v.x, (float)v.y, (float)v.z);
}

// Other overloads for float3

inline __host__ __device__ float3 clamp(const float3 &v, const float min, const float max) {
    return make_float3(
        fminf(max, fmaxf(min, v.x)),
        fminf(max, fmaxf(min, v.y)),
        fminf(max, fmaxf(min, v.z))
    );
}

inline __device__ int mip_from_pos(const float3 &v, const float max_cascade) {
    return mip_from_pos(v.x, v.y, v.z, max_cascade);
}

inline __host__ __device__ uint32_t __morton3D(uint3 v) {
    return __morton3D(v.x, v.y, v.z);
}

inline __device__ float3 signf(const float3 &v) {
    return make_float3(signf(v.x), signf(v.y), signf(v.z));
}

////////////////////////////////////////////////////
/////////////           utils          /////////////
////////////////////////////////////////////////////

// rays_o/d: [N, 3]
// nears/fars: [N]
// scalar_t should always be float in use.
template <typename scalar_t>
__global__ void kernel_near_far_from_aabb(
    const scalar_t * __restrict__ rays_o,
    const scalar_t * __restrict__ rays_d,
    const scalar_t * __restrict__ aabb,
    const uint32_t N,
    const float min_near,
    scalar_t * nears, scalar_t * fars
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    rays_o += n * 3;
    rays_d += n * 3;

    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;

    // get near far (assume cube scene)
    float near = (aabb[0] - ox) * rdx;
    float far = (aabb[3] - ox) * rdx;
    if (near > far) swapf(near, far);

    float near_y = (aabb[1] - oy) * rdy;
    float far_y = (aabb[4] - oy) * rdy;
    if (near_y > far_y) swapf(near_y, far_y);

    if (near > far_y || near_y > far) {
        nears[n] = fars[n] = std::numeric_limits<scalar_t>::max();
        return;
    }

    if (near_y > near) near = near_y;
    if (far_y < far) far = far_y;

    float near_z = (aabb[2] - oz) * rdz;
    float far_z = (aabb[5] - oz) * rdz;
    if (near_z > far_z) swapf(near_z, far_z);

    if (near > far_z || near_z > far) {
        nears[n] = fars[n] = std::numeric_limits<scalar_t>::max();
        return;
    }

    if (near_z > near) near = near_z;
    if (far_z < far) far = far_z;

    if (near < min_near) near = min_near;

    nears[n] = near;
    fars[n] = far;
}


void near_far_from_aabb(const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor aabb, const uint32_t N, const float min_near, at::Tensor nears, at::Tensor fars) {

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "near_far_from_aabb", ([&] {
        kernel_near_far_from_aabb<<<div_round_up(N, N_THREAD), N_THREAD>>>(rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), aabb.data_ptr<scalar_t>(), N, min_near, nears.data_ptr<scalar_t>(), fars.data_ptr<scalar_t>());
    }));
}


// rays_o/d: [N, 3]
// radius: float
// coords: [N, 2]
template <typename scalar_t>
__global__ void kernel_sph_from_ray(
    const scalar_t * __restrict__ rays_o,
    const scalar_t * __restrict__ rays_d,
    const float radius,
    const uint32_t N,
    scalar_t * coords
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    rays_o += n * 3;
    rays_d += n * 3;
    coords += n * 2;

    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;

    // solve t from || o + td || = radius
    const float A = dx * dx + dy * dy + dz * dz;
    const float B = ox * dx + oy * dy + oz * dz; // in fact B / 2
    const float C = ox * ox + oy * oy + oz * oz - radius * radius;

    const float t = (- B + sqrtf(B * B - A * C)) / A; // always use the larger solution (positive)

    // solve theta, phi (assume y is the up axis)
    const float x = ox + t * dx, y = oy + t * dy, z = oz + t * dz;
    const float theta = atan2(sqrtf(x * x + z * z), y); // [0, PI)
    const float phi = atan2(z, x); // [-PI, PI)

    // normalize to [-1, 1]
    coords[0] = 2 * theta * RPI() - 1;
    coords[1] = phi * RPI();
}


void sph_from_ray(const at::Tensor rays_o, const at::Tensor rays_d, const float radius, const uint32_t N, at::Tensor coords) {

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "sph_from_ray", ([&] {
        kernel_sph_from_ray<<<div_round_up(N, N_THREAD), N_THREAD>>>(rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), radius, N, coords.data_ptr<scalar_t>());
    }));
}


// coords: int32, [N, 3]
// indices: int32, [N]
__global__ void kernel_morton3D(
    const int * __restrict__ coords,
    const uint32_t N,
    int * indices
) {
    // parallel
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    coords += n * 3;
    indices[n] = __morton3D(coords[0], coords[1], coords[2]);
}


void morton3D(const at::Tensor coords, const uint32_t N, at::Tensor indices) {
    static constexpr uint32_t N_THREAD = 128;
    kernel_morton3D<<<div_round_up(N, N_THREAD), N_THREAD>>>(coords.data_ptr<int>(), N, indices.data_ptr<int>());
}


// indices: int32, [N]
// coords: int32, [N, 3]
__global__ void kernel_morton3D_invert(
    const int * __restrict__ indices,
    const uint32_t N,
    int * coords
) {
    // parallel
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    coords += n * 3;

    const int ind = indices[n];

    coords[0] = __morton3D_invert(ind >> 0);
    coords[1] = __morton3D_invert(ind >> 1);
    coords[2] = __morton3D_invert(ind >> 2);
}


void morton3D_invert(const at::Tensor indices, const uint32_t N, at::Tensor coords) {
    static constexpr uint32_t N_THREAD = 128;
    kernel_morton3D_invert<<<div_round_up(N, N_THREAD), N_THREAD>>>(indices.data_ptr<int>(), N, coords.data_ptr<int>());
}


// grid: float, [C, H, H, H]
// N: int, C * H * H * H / 8
// density_thresh: float
// bitfield: uint8, [N]
template <typename scalar_t>
__global__ void kernel_packbits(
    const scalar_t * __restrict__ grid,
    const uint32_t N,
    const float density_thresh,
    uint8_t * bitfield
) {
    // parallel per byte
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    grid += n * 8;

    uint8_t bits = 0;

    #pragma unroll
    for (uint8_t i = 0; i < 8; i++) {
        bits |= (grid[i] > density_thresh) ? ((uint8_t)1 << i) : 0;
    }

    bitfield[n] = bits;
}


void packbits(const at::Tensor grid, const uint32_t N, const float density_thresh, at::Tensor bitfield) {

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grid.scalar_type(), "packbits", ([&] {
        kernel_packbits<<<div_round_up(N, N_THREAD), N_THREAD>>>(grid.data_ptr<scalar_t>(), N, density_thresh, bitfield.data_ptr<uint8_t>());
    }));
}

////////////////////////////////////////////////////
/////////////         training         /////////////
////////////////////////////////////////////////////

// rays_o/d: [N, 3]
// grid: [CHHH / 8]
// xyzs, dirs, deltas: [M, 3], [M, 3], [M, 2]
// dirs: [M, 3]
// rays: [N, 3], idx, offset, num_steps
template <typename scalar_t>
__global__ void kernel_march_rays_train(
    const scalar_t * __restrict__ rays_o,
    const scalar_t * __restrict__ rays_d,  
    const scalar_t * __restrict__ z_hats,
    const uint8_t * __restrict__ grid,
    const float bound,
    const float dt_gamma, const uint32_t max_steps, const bool is_ndc,
    const uint32_t N, const uint32_t C, const uint32_t H, const uint32_t M,
    const scalar_t* __restrict__ nears, 
    const scalar_t* __restrict__ fars,
    scalar_t * xyzs, scalar_t * dirs, scalar_t * deltas,
    int * rays,
    int * counter,
    const scalar_t* __restrict__ noises
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    rays_o += n * 3;
    rays_d += n * 3;
    z_hats += n;

    // ray marching
    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;
    const float rH = 1 / (float)H;
    const float H3 = H * H * H;

    const float near = nears[n];
    const float far = fars[n];
    const float noise = noises[n];

    const float dt_min = 2 * SQRT3() / max_steps;
    const float dt_max = 2 * SQRT3() * (1 << (C - 1)) / H;
    
    float t0 = near;
    
    // perturb
    t0 += clamp(t0 * dt_gamma, dt_min, dt_max) * noise;

    // first pass: estimation of num_steps
    float t = t0;
    uint32_t num_steps = 0;

    //if (t < far) printf("valid ray %d t=%f near=%f far=%f \n", n, t, near, far);
    
    while (t < far && num_steps < max_steps) {
        // current point
        const float x = clamp(ox + t * dx, -bound, bound);
        const float y = clamp(oy + t * dy, -bound, bound);
        const float z = clamp(oz + t * dz, -bound, bound);

        const float dt = clamp(t * dt_gamma, dt_min, dt_max);

        // get mip level
        const int level = max(mip_from_pos(x, y, z, C), mip_from_dt(dt, H, C)); // range in [0, C - 1]

        const float mip_bound = fminf(scalbnf(1.0f, level), bound);
        const float mip_rbound = 1 / mip_bound;
        
        // convert to nearest grid position
        const int nx = clamp(0.5 * (x * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int ny = clamp(0.5 * (y * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int nz = clamp(0.5 * (z * mip_rbound + 1) * H, 0.0f, (float)(H - 1));

        const uint32_t index = level * H3 + __morton3D(nx, ny, nz);
        const bool occ = grid[index / 8] & (1 << (index % 8));

        // if occpuied, advance a small step, and write to output
        //if (n == 0) printf("t=%f density=%f vs thresh=%f step=%d\n", t, density, density_thresh, num_steps);

        if (occ) {
            num_steps++;
            t += dt;
        // else, skip a large step (basically skip a voxel grid)
        } else {
            // calc distance to next voxel
            const float tx = (((nx + 0.5f + 0.5f * signf(dx)) * rH * 2 - 1) * mip_bound - x) * rdx;
            const float ty = (((ny + 0.5f + 0.5f * signf(dy)) * rH * 2 - 1) * mip_bound - y) * rdy;
            const float tz = (((nz + 0.5f + 0.5f * signf(dz)) * rH * 2 - 1) * mip_bound - z) * rdz;

            const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do { 
                t += clamp(t * dt_gamma, dt_min, dt_max);
            } while (t < tt);
        }
    }

    //printf("[n=%d] num_steps=%d, near=%f, far=%f, dt=%f, max_steps=%f\n", n, num_steps, near, far, dt_min, (far - near) / dt_min);

    // second pass: really locate and write points & dirs
    uint32_t point_index = atomicAdd(counter, num_steps);
    uint32_t ray_index = atomicAdd(counter + 1, 1);
    
    //printf("[n=%d] num_steps=%d, point_index=%d, ray_index=%d\n", n, num_steps, point_index, ray_index);

    // write rays
    rays[ray_index * 3] = n;
    rays[ray_index * 3 + 1] = point_index;
    rays[ray_index * 3 + 2] = num_steps;

    if (num_steps == 0) return;
    if (point_index + num_steps >= M) return;

    xyzs += point_index * 3;
    dirs += point_index * 3;
    deltas += point_index * 4;

    t = t0;
    uint32_t step = 0;

    float last_t = t;
    float last_z = clamp(oz + t * dz, -bound, bound);
    float new_z;

    while (t < far && step < num_steps) {
        // current point
        const float x = clamp(ox + t * dx, -bound, bound);
        const float y = clamp(oy + t * dy, -bound, bound);
        const float z = clamp(oz + t * dz, -bound, bound);

        const float dt = clamp(t * dt_gamma, dt_min, dt_max);

        // get mip level
        const int level = max(mip_from_pos(x, y, z, C), mip_from_dt(dt, H, C)); // range in [0, C - 1]

        const float mip_bound = fminf(scalbnf(1.0f, level), bound);
        const float mip_rbound = 1 / mip_bound;
        
        // convert to nearest grid position
        const int nx = clamp(0.5 * (x * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int ny = clamp(0.5 * (y * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int nz = clamp(0.5 * (z * mip_rbound + 1) * H, 0.0f, (float)(H - 1));

        // query grid
        const uint32_t index = level * H3 + __morton3D(nx, ny, nz);
        const bool occ = grid[index / 8] & (1 << (index % 8));

        // if occpuied, advance a small step, and write to output
        if (occ) {
            // write step
            xyzs[0] = x;
            xyzs[1] = y;
            xyzs[2] = z;
            dirs[0] = dx;
            dirs[1] = dy;
            dirs[2] = dz;
            t += dt;
            deltas[0] = dt;
            deltas[1] = t - last_t; // used to calc depth
            last_t = t;
            if (is_ndc) {
                new_z = clamp(oz + t * dz, -bound, bound);
                deltas[2] = (2 / (new_z - 1) - 2 / (z - 1)) / z_hats[0];
                deltas[3] = (2 / (new_z - 1) - 2 / (last_z - 1)) / z_hats[0];
                last_z = z;
            }
            xyzs += 3;
            dirs += 3;
            deltas += 4;
            step++;
        // else, skip a large step (basically skip a voxel grid)
        } else {
            // calc distance to next voxel
            const float tx = (((nx + 0.5f + 0.5f * signf(dx)) * rH * 2 - 1) * mip_bound - x) * rdx;
            const float ty = (((ny + 0.5f + 0.5f * signf(dy)) * rH * 2 - 1) * mip_bound - y) * rdy;
            const float tz = (((nz + 0.5f + 0.5f * signf(dz)) * rH * 2 - 1) * mip_bound - z) * rdz;
            const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do { 
                t += clamp(t * dt_gamma, dt_min, dt_max); 
            } while (t < tt);
        }
    }
}

void march_rays_train(const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor z_hats, const at::Tensor grid, const float bound, const float dt_gamma, const uint32_t max_steps, const bool is_ndc, const uint32_t N, const uint32_t C, const uint32_t H, const uint32_t M, const at::Tensor nears, const at::Tensor fars, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor rays, at::Tensor counter, at::Tensor noises) {

    static constexpr uint32_t N_THREAD = 128;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "march_rays_train", ([&] {
        kernel_march_rays_train<<<div_round_up(N, N_THREAD), N_THREAD>>>(rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), z_hats.data_ptr<scalar_t>(), grid.data_ptr<uint8_t>(), bound, dt_gamma, max_steps, is_ndc, N, C, H, M, nears.data_ptr<scalar_t>(), fars.data_ptr<scalar_t>(), xyzs.data_ptr<scalar_t>(), dirs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), rays.data_ptr<int>(), counter.data_ptr<int>(), noises.data_ptr<scalar_t>());
    }));
}


template <typename scalar_t>
__global__ void kernel_march_rays_unbounded_train(
    const scalar_t * __restrict__ rays_o,
    const scalar_t * __restrict__ rays_d,
    const uint8_t * __restrict__ grid,
    const float bound,
    const float dt_gamma,
    const uint32_t max_steps,
    const uint32_t N,
    const uint32_t C,
    const uint32_t H,
    const uint32_t M,
    const scalar_t * __restrict__ nears,
    const scalar_t * __restrict__ fars,
    scalar_t * xyzs,
    scalar_t * dirs,
    scalar_t * deltas,
    int * rays,
    int * counter,
    const scalar_t * __restrict__ noises
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N)
        return;
    
    rays_o += n * 3;
    rays_d += n * 3;

    const float3 o = make_float3(rays_o[0], rays_o[1], rays_o[2]);
    const float3 d = make_float3(rays_d[0], rays_d[1], rays_d[2]);
    const float3 rd = 1 / d;

    const float rH = 1 / (float)H;
    const float H3 = H * H * H;

    const float near = nears[n];
    const float far = fars[n];
    const float noise = noises[n];

    const float dt_min = 2 * SQRT3() / max_steps;
    const float dt_max = 2 * SQRT3() * (1 << (C - 1)) / H;

    float t0 = near;
    t0 += clamp(t0 * dt_gamma, dt_min, dt_max) * noise;

    float t = t0;
    uint32_t num_steps = 0;

    while (t < far && num_steps < max_steps) {
        const float3 p = clamp(o + t * d, -bound, bound);
        const float dt = clamp(t * dt_gamma, dt_min, dt_max);

        const int level = max(mip_from_pos(p, C), mip_from_dt(dt, H, C)); // [0, C)
        const float mip_bound = fminf(scalbnf(1.0f, level), bound); // min(2^lvl, bound)
        const float mip_rbound = 1 / mip_bound;

        // corresponding cell for p
        const uint3 np = to_uint3(clamp(0.5 * (p * mip_rbound + 1) * H, 0.0f, (float)(H - 1)));
        const uint32_t index = level * H3 + __morton3D(np);
        const bool occ = grid[index / 8] & (1 << (index % 8));

        if (occ) {
            num_steps++;
            t += dt;
        } else {
            const float3 next_ts = ((
                (to_float3(np) + 0.5f + 0.5f * signf(d)) * rH * 2 - 1) * mip_bound - p) * rd;
            const float tt = t + fmaxf(0.0f, fminf(next_ts));

            do {
                t += clamp(t * dt_gamma, dt_min, dt_max);
            } while (t < tt);
        }
    }

    float r0 = 1 / norm(clamp(o + t * d, -bound, bound));
    float r = r0;
    float thres_r = 0.05;  // closer to 0 -> render into infinity
    float dr = 0.005;

    const float tb = -dot(d, o);  // assuming |d| = 1
    const float b = norm(o + tb * d);  // B = closest pt on ray to origin; refer to NeRF++ paper

    while (r > thres_r) {
        r -= dr;
        const float tr = sqrtf(1 / (r * r) - b * b) + tb;
        const float3 p = o + tr * d;  // point in Euclidean space
        const float3 pinv = p * r * r;  // point in inverted sphere space

        const uint3 np = to_uint3(clamp(0.5 * (p + 1) * H, 0.0f, (float)(H - 1)));
        const uint32_t index = C * H3 + __morton3D(np);
        const bool occ = grid[index / 8] & (1 << (index % 8));
        
        if (occ) {
            num_steps++;
        }
    }

    uint32_t point_index = atomicAdd(counter, num_steps);
    uint32_t ray_index = atomicAdd(counter + 1, 1);

    rays[ray_index * 3] = n;
    rays[ray_index * 3 + 1] = point_index;
    rays[ray_index * 3 + 2] = num_steps;

    return;

    if ((num_steps == 0) || (point_index + num_steps >= M))
        return;

    xyzs += point_index * 3;
    dirs += point_index * 3;
    deltas += point_index * 4;

    t = t0;
    uint32_t step = 0;
    float last_t = t;

    while (t < far && step < num_steps) {
        const float3 p = clamp(o + t * d, -bound, bound);
        const float dt = clamp(t * dt_gamma, dt_min, dt_max);

        const int level = max(mip_from_pos(p, C), mip_from_dt(dt, H, C)); // [0, C)
        const float mip_bound = fminf(scalbnf(1.0f, level), bound); // min(2^lvl, bound)
        const float mip_rbound = 1 / mip_bound;

        // corresponding cell for p
        const uint3 np = to_uint3(clamp(0.5 * (p * mip_rbound + 1) * H, 0.0f, (float)(H - 1)));
        const uint32_t index = level * H3 + __morton3D(np);
        const bool occ = grid[index / 8] & (1 << (index % 8));

        if (occ) {
            write_float3(p, xyzs);
            write_float3(d, dirs);
            t += dt;
            deltas[0] = dt;
            deltas[1] = t - last_t;
            last_t = t;
            xyzs += 3;
            dirs += 3;
            step++;
        } else {
            const float3 next_ts = ((
                (to_float3(np) + 0.5f + 0.5f * signf(d)) * rH * 2 - 1) * mip_bound - p) * rd;
            const float tt = t + fmaxf(0.0f, fminf(next_ts));

            do {
                t += clamp(t * dt_gamma, dt_min, dt_max);
            } while (t < tt);
        }
    }
}

void march_rays_unbounded_train(
    const at::Tensor rays_o,
    const at::Tensor rays_d,
    const at::Tensor grid,
    const float bound,
    const float dt_gamma,
    const uint32_t max_steps,
    const uint32_t N,
    const uint32_t C,
    const uint32_t H,
    const uint32_t M,
    const at::Tensor nears,
    const at::Tensor fars,
    at::Tensor xyzs,
    at::Tensor dirs,
    at::Tensor deltas,
    at::Tensor rays,
    at::Tensor counter,
    at::Tensor noises
) {
    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        rays_o.scalar_type(),
        "march_rays_train",
        ([&] {
            kernel_march_rays_unbounded_train<<<div_round_up(N, N_THREAD), N_THREAD>>>(
                rays_o.data_ptr<scalar_t>(),
                rays_d.data_ptr<scalar_t>(),
                grid.data_ptr<uint8_t>(),
                bound, dt_gamma, max_steps, N, C, H, M,
                nears.data_ptr<scalar_t>(),
                fars.data_ptr<scalar_t>(),
                xyzs.data_ptr<scalar_t>(),
                dirs.data_ptr<scalar_t>(),
                deltas.data_ptr<scalar_t>(),
                rays.data_ptr<int>(),
                counter.data_ptr<int>(),
                noises.data_ptr<scalar_t>()
            );
        })
    );
}

// sigmas: [M]
// rgbs: [M, C]
// deltas: [M, 2]
// rays: [N, 3], idx, offset, num_steps
// weights_sum: [N], final pixel alpha
// depth: [N,]
// image: [N, C]
template <typename scalar_t>
__global__ void kernel_composite_rays_train_forward(
    const scalar_t * __restrict__ sigmas,
    const scalar_t * __restrict__ rgbs,  
    const scalar_t * __restrict__ deltas,
    const int * __restrict__ rays,
    const uint32_t M, const uint32_t N, const uint32_t C,
    const float T_thresh, const bool is_ndc,
    scalar_t * weights_sum,
    scalar_t * depth,
    scalar_t * image
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate 
    uint32_t index = rays[n * 3];
    uint32_t offset = rays[n * 3 + 1];
    uint32_t num_steps = rays[n * 3 + 2];

    // empty ray, or ray that exceed max step count.
    for (int i = 0; i < C; ++i)
        image[index * C + i] = 0;
    if (num_steps == 0 || offset + num_steps >= M) {
        weights_sum[index] = 0;
        depth[index] = 0;
        return;
    }

    sigmas += offset;
    rgbs += offset * C;
    deltas += offset * 4;

    // accumulate 
    uint32_t step = 0;

    scalar_t T = 1.0f;
    scalar_t ws = 0, t = 0, d = 0;

    while (step < num_steps) {

        const scalar_t alpha = 1.0f - __expf(- sigmas[0] * (is_ndc ? deltas[2] : deltas[0]));
        const scalar_t weight = alpha * T;

        for (int i = 0; i < C; ++i)
            image[index * C + i] += weight * rgbs[i];
        
        t += (is_ndc ? deltas[3] : deltas[1]); // real delta
        d += weight * t;
        
        ws += weight;
        
        T *= 1.0f - alpha;

        // minimal remained transmittence
        if (T < T_thresh) break;

        //printf("[n=%d] num_steps=%d, alpha=%f, w=%f, T=%f, sum_dt=%f, d=%f\n", n, step, alpha, weight, T, sum_delta, d);

        // locate
        sigmas++;
        rgbs += C;
        deltas += 4;

        step++;
    }

    //printf("[n=%d] rgb=(%f, %f, %f), d=%f\n", n, r, g, b, d);

    // write
    weights_sum[index] = ws; // weights_sum
    depth[index] = d;
}


void composite_rays_train_forward(const at::Tensor sigmas, const at::Tensor rgbs, const at::Tensor deltas, const at::Tensor rays, const uint32_t M, const uint32_t N, const uint32_t C, const float T_thresh, const bool is_ndc, at::Tensor weights_sum, at::Tensor depth, at::Tensor image) {

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    sigmas.scalar_type(), "composite_rays_train_forward", ([&] {
        kernel_composite_rays_train_forward<<<div_round_up(N, N_THREAD), N_THREAD>>>(sigmas.data_ptr<scalar_t>(), rgbs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), rays.data_ptr<int>(), M, N, C, T_thresh, is_ndc, weights_sum.data_ptr<scalar_t>(), depth.data_ptr<scalar_t>(), image.data_ptr<scalar_t>());
    }));
}


// grad_weights_sum: [N,]
// grad: [N, C]
// sigmas: [M]
// rgbs: [M, C]
// deltas: [M, 2]
// rays: [N, 3], idx, offset, num_steps
// weights_sum: [N,], weights_sum here 
// image: [N, C]
// grad_sigmas: [M]
// grad_rgbs: [M, C]
// rgbs_buf: [N, C]
template <typename scalar_t>
__global__ void kernel_composite_rays_train_backward(
    const scalar_t * __restrict__ grad_weights_sum,
    const scalar_t * __restrict__ grad_image,
    const scalar_t * __restrict__ sigmas,
    const scalar_t * __restrict__ rgbs, 
    const scalar_t * __restrict__ deltas,
    const int * __restrict__ rays,
    const bool is_ndc,
    const scalar_t * __restrict__ weights_sum,
    const scalar_t * __restrict__ image,
    const uint32_t M, const uint32_t N, const uint32_t C, const float T_thresh,
    scalar_t * grad_sigmas,
    scalar_t * grad_rgbs,
    scalar_t * rgbs_buf
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate 
    uint32_t index = rays[n * 3];
    uint32_t offset = rays[n * 3 + 1];
    uint32_t num_steps = rays[n * 3 + 2];

    if (num_steps == 0 || offset + num_steps >= M) return;

    grad_weights_sum += index;
    grad_image += index * C;
    weights_sum += index;
    image += index * C;
    sigmas += offset;
    rgbs += offset * C;
    deltas += offset * 4;
    grad_sigmas += offset;
    grad_rgbs += offset * C;
    rgbs_buf += index * C;

    // accumulate 
    uint32_t step = 0;
    
    scalar_t T = 1.0f;
    const scalar_t ws_final = weights_sum[0];
    scalar_t ws = 0;

    while (step < num_steps) {
        
        const scalar_t alpha = 1.0f - __expf(- sigmas[0] * (is_ndc ? deltas[2] : deltas[0]));
        const scalar_t weight = alpha * T;

        for (int i = 0; i < C; ++i)
            rgbs_buf[i] += weight * rgbs[i];
        ws += weight;

        T *= 1.0f - alpha;

        // minimal remained transmittence
        if (T < T_thresh) break;

        // check https://note.kiui.moe/others/nerf_gradient/ for the gradient calculation.
        // write grad_rgbs
        for (int i = 0; i < C; ++i)
            grad_rgbs[i] = grad_image[i] * weight;

        // write grad_sigmas
        scalar_t grad_image_sum = 0;
        for (int i = 0; i < C; ++i)
            grad_image_sum += (grad_image[i] * (T * rgbs[i] - (image[i] - rgbs_buf[i])));
        grad_sigmas[0] = (is_ndc ? deltas[2] : deltas[0]) * (
            grad_image_sum + grad_weights_sum[0] * (1 - ws_final));

        //printf("[n=%d] num_steps=%d, T=%f, grad_sigmas=%f, r_final=%f, r=%f\n", n, step, T, grad_sigmas[0], r_final, r);
    
        // locate
        sigmas++;
        rgbs += C;
        deltas += 4;
        grad_sigmas++;
        grad_rgbs += C;

        step++;
    }
}


void composite_rays_train_backward(const at::Tensor grad_weights_sum, const at::Tensor grad_image, const at::Tensor sigmas, const at::Tensor rgbs, const at::Tensor deltas, const at::Tensor rays, const bool is_ndc, const at::Tensor weights_sum, const at::Tensor image, const uint32_t M, const uint32_t N, const uint32_t C, const float T_thresh, at::Tensor grad_sigmas, at::Tensor grad_rgbs, at::Tensor rgbs_buf) {

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad_image.scalar_type(), "composite_rays_train_backward", ([&] {
        kernel_composite_rays_train_backward<<<div_round_up(N, N_THREAD), N_THREAD>>>(grad_weights_sum.data_ptr<scalar_t>(), grad_image.data_ptr<scalar_t>(), sigmas.data_ptr<scalar_t>(), rgbs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), rays.data_ptr<int>(), is_ndc, weights_sum.data_ptr<scalar_t>(), image.data_ptr<scalar_t>(), M, N, C, T_thresh, grad_sigmas.data_ptr<scalar_t>(), grad_rgbs.data_ptr<scalar_t>(), rgbs_buf.data_ptr<scalar_t>());
    }));
}


////////////////////////////////////////////////////
/////////////          infernce        /////////////
////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void kernel_march_rays(
    const uint32_t n_alive, 
    const uint32_t n_step, 
    const int* __restrict__ rays_alive, 
    const scalar_t* __restrict__ rays_t, 
    const scalar_t* __restrict__ rays_o, 
    const scalar_t* __restrict__ rays_d, 
    const scalar_t* __restrict__ z_hats,
    const float bound,
    const float dt_gamma, const uint32_t max_steps, const bool is_ndc,
    const uint32_t C, const uint32_t H,
    const uint8_t * __restrict__ grid,
    const scalar_t* __restrict__ nears,
    const scalar_t* __restrict__ fars,
    scalar_t* xyzs, scalar_t* dirs, scalar_t* deltas,
    const scalar_t* __restrict__ noises
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= n_alive) return;

    const int index = rays_alive[n]; // ray id
    const float noise = noises[n];
    
    // locate
    rays_o += index * 3;
    rays_d += index * 3;
    rays_t += index * (is_ndc ? 2 : 1);
    z_hats += index;
    xyzs += n * n_step * 3;
    dirs += n * n_step * 3;
    deltas += n * n_step * 4;
    
    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;
    const float rH = 1 / (float)H;
    const float H3 = H * H * H;
    
    float t = rays_t[0]; // current ray's t
    const float near = nears[index], far = fars[index];

    const float dt_min = 2 * SQRT3() / max_steps;
    const float dt_max = 2 * SQRT3() * (1 << (C - 1)) / H;

    // march for n_step steps, record points
    uint32_t step = 0;

    // introduce some randomness
    t += clamp(t * dt_gamma, dt_min, dt_max) * noise;

    float last_t = t;
    float last_z = clamp(oz + t * dz, -bound, bound);
    float new_z;

    while (t < far && step < n_step) {
        // current point
        const float x = clamp(ox + t * dx, -bound, bound);
        const float y = clamp(oy + t * dy, -bound, bound);
        const float z = clamp(oz + t * dz, -bound, bound);

        const float dt = clamp(t * dt_gamma, dt_min, dt_max);

        // get mip level
        const int level = max(mip_from_pos(x, y, z, C), mip_from_dt(dt, H, C)); // range in [0, C - 1]

        const float mip_bound = fminf(scalbnf(1, level), bound);
        const float mip_rbound = 1 / mip_bound;
        
        // convert to nearest grid position
        const int nx = clamp(0.5 * (x * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int ny = clamp(0.5 * (y * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int nz = clamp(0.5 * (z * mip_rbound + 1) * H, 0.0f, (float)(H - 1));

        const uint32_t index = level * H3 + __morton3D(nx, ny, nz);
        const bool occ = grid[index / 8] & (1 << (index % 8));

        // if occpuied, advance a small step, and write to output
        if (occ) {
            // write step
            xyzs[0] = x;
            xyzs[1] = y;
            xyzs[2] = z;
            dirs[0] = dx;
            dirs[1] = dy;
            dirs[2] = dz;
            // calc dt
            t += dt;
            deltas[0] = dt;
            deltas[1] = t - last_t; // used to calc depth
            if (is_ndc) {
                new_z = clamp(oz + t * dz, -bound, bound);
                deltas[2] = (2 / (new_z - 1) - 2 / (z - 1)) / z_hats[0];
                deltas[3] = (2 / (new_z - 1) - 2 / (last_z - 1)) / z_hats[0];
                last_z = new_z;
            }
            last_t = t;
            // step
            xyzs += 3;
            dirs += 3;
            deltas += 4;
            step++;

        // else, skip a large step (basically skip a voxel grid)
        } else {
            // calc distance to next voxel
            const float tx = (((nx + 0.5f + 0.5f * signf(dx)) * rH * 2 - 1) * mip_bound - x) * rdx;
            const float ty = (((ny + 0.5f + 0.5f * signf(dy)) * rH * 2 - 1) * mip_bound - y) * rdy;
            const float tz = (((nz + 0.5f + 0.5f * signf(dz)) * rH * 2 - 1) * mip_bound - z) * rdz;
            const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do { 
                t += clamp(t * dt_gamma, dt_min, dt_max);
            } while (t < tt);
        }
    }
}


void march_rays(const uint32_t n_alive, const uint32_t n_step, const at::Tensor rays_alive, const at::Tensor rays_t, const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor z_hats, const float bound, const float dt_gamma, const uint32_t max_steps, const bool is_ndc, const uint32_t C, const uint32_t H, const at::Tensor grid, const at::Tensor near, const at::Tensor far, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor noises) {
    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "march_rays", ([&] {
        kernel_march_rays<<<div_round_up(n_alive, N_THREAD), N_THREAD>>>(n_alive, n_step, rays_alive.data_ptr<int>(), rays_t.data_ptr<scalar_t>(), rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), z_hats.data_ptr<scalar_t>(), bound, dt_gamma, max_steps, is_ndc, C, H, grid.data_ptr<uint8_t>(), near.data_ptr<scalar_t>(), far.data_ptr<scalar_t>(), xyzs.data_ptr<scalar_t>(), dirs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), noises.data_ptr<scalar_t>());
    }));
}


template <typename scalar_t>
__global__ void kernel_composite_rays(
    const uint32_t n_alive, 
    const uint32_t n_step, 
    const float T_thresh,
    int* rays_alive, 
    scalar_t* rays_t, 
    const scalar_t* __restrict__ sigmas, 
    const scalar_t* __restrict__ rgbs, 
    const scalar_t* __restrict__ deltas, 
    const uint32_t C,
    const bool is_ndc,
    scalar_t* weights_sum, scalar_t* depth, scalar_t* image
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= n_alive) return;

    const int index = rays_alive[n]; // ray id
    
    // locate 
    sigmas += n * n_step;
    rgbs += n * n_step * C;
    deltas += n * n_step * 4;
    
    rays_t += index * (is_ndc ? 2 : 1);
    weights_sum += index;
    depth += index;
    image += index * C;

    scalar_t t_rm, t_phy;
    if (is_ndc) {
        t_rm = rays_t[0];
        t_phy = rays_t[1];
    } else {
        t_phy = rays_t[0];
    }
    
    scalar_t weight_sum = weights_sum[0];
    scalar_t d = depth[0];

    // accumulate 
    uint32_t step = 0;
    while (step < n_step) {
        
        // ray is terminated if delta == 0
        if (deltas[0] == 0) break;
        
        const scalar_t alpha = 1.0f - __expf(- sigmas[0] * (is_ndc ? deltas[2] : deltas[0]));

        /* 
        T_0 = 1; T_i = \prod_{j=0}^{i-1} (1 - alpha_j)
        w_i = alpha_i * T_i
        --> 
        T_i = 1 - \sum_{j=0}^{i-1} w_j
        */
        const scalar_t T = 1 - weight_sum;
        const scalar_t weight = alpha * T;
        weight_sum += weight;

        if (is_ndc) {
            t_rm += deltas[1];
            t_phy += deltas[3];
        } else {
            t_phy += deltas[1];
        }
        d += weight * t_phy;
        for (int i = 0; i < C; ++i)
            image[i] += weight * rgbs[i];

        //printf("[n=%d] num_steps=%d, alpha=%f, w=%f, T=%f, sum_dt=%f, d=%f\n", n, step, alpha, weight, T, sum_delta, d);

        // ray is terminated if T is too small
        // use a larger bound to further accelerate inference
        if (T < T_thresh) break;

        // locate
        sigmas++;
        rgbs += C;
        deltas += 4;
        step++;
    }

    //printf("[n=%d] rgb=(%f, %f, %f), d=%f\n", n, r, g, b, d);

    // rays_alive = -1 means ray is terminated early.
    if (step < n_step) {
        rays_alive[n] = -1;
    } else {
        if (is_ndc) {
            rays_t[0] = t_rm;
            rays_t[1] = t_phy;
        } else {
            rays_t[0] = t_phy;
        }
    }

    weights_sum[0] = weight_sum; // this is the thing I needed!
    depth[0] = d;
}


void composite_rays(const uint32_t n_alive, const uint32_t n_step, const float T_thresh, at::Tensor rays_alive, at::Tensor rays_t, const at::Tensor sigmas, const at::Tensor rgbs, const at::Tensor deltas, const uint32_t C, const bool is_ndc, at::Tensor weights, at::Tensor depth, at::Tensor image) {
    static constexpr uint32_t N_THREAD = 128;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    image.scalar_type(), "composite_rays", ([&] {
        kernel_composite_rays<<<div_round_up(n_alive, N_THREAD), N_THREAD>>>(n_alive, n_step, T_thresh, rays_alive.data_ptr<int>(), rays_t.data_ptr<scalar_t>(), sigmas.data_ptr<scalar_t>(), rgbs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), C, is_ndc, weights.data_ptr<scalar_t>(), depth.data_ptr<scalar_t>(), image.data_ptr<scalar_t>());
    }));
}