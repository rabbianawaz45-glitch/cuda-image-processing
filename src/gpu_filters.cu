// gpu_filters.cu -- CUDA kernels + host wrappers.
// Grayscale + timing helper: Member A | Blur (global + shared): Member C | Sobel: Member D
// One thread = one output pixel, 2D grid of 16x16 blocks.
#include "common.h"
#include "cuda_utils.h"

#define BLOCK 16
#define TILE  BLOCK   // shared-memory tile is TILE x TILE plus a 1-pixel halo

__device__ __forceinline__ int clampi_d(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// ---------------------------------------------------------------- Grayscale (Member A)
__global__ void grayscaleKernel(const unsigned char* rgb, unsigned char* gray, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;                       // boundary check
    int i = y * w + x;
    float g = 0.299f * rgb[3 * i] + 0.587f * rgb[3 * i + 1] + 0.114f * rgb[3 * i + 2];
    gray[i] = (unsigned char)(g + 0.5f);
}

// ---------------------------------------------------------------- Blur, global memory (Member C)
__global__ void blurGlobalKernel(const unsigned char* in, unsigned char* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int sum = 0;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
            sum += in[clampi_d(y + dy, 0, h - 1) * w + clampi_d(x + dx, 0, w - 1)];
    out[y * w + x] = (unsigned char)((sum + 4) / 9);
}

// ---------------------------------------------------------------- Blur, shared memory (Member C)
// Launch with blockDim = (TILE, TILE). Each block loads an (TILE+2)x(TILE+2) tile (with halo).
__global__ void blurSharedKernel(const unsigned char* in, unsigned char* out, int w, int h) {
    __shared__ unsigned char tile[TILE + 2][TILE + 2];
    int tx = threadIdx.x, ty = threadIdx.y;
    int x = blockIdx.x * TILE + tx;
    int y = blockIdx.y * TILE + ty;

    // ALL threads take part in loading (even if outside the image) -> no deadlock at the barrier
    for (int dy = ty; dy < TILE + 2; dy += TILE) {
        for (int dx = tx; dx < TILE + 2; dx += TILE) {
            int gx = clampi_d(blockIdx.x * TILE + dx - 1, 0, w - 1);
            int gy = clampi_d(blockIdx.y * TILE + dy - 1, 0, h - 1);
            tile[dy][dx] = in[gy * w + gx];
        }
    }
    __syncthreads();                                     // tile fully loaded before anyone reads it

    if (x < w && y < h) {                                // only now skip out-of-image threads
        int sum = 0;
        for (int j = 0; j < 3; j++)
            for (int i = 0; i < 3; i++)
                sum += tile[ty + j][tx + i];
        out[y * w + x] = (unsigned char)((sum + 4) / 9);
    }
}

// ---------------------------------------------------------------- Sobel (Member D)
__global__ void sobelKernel(const unsigned char* in, unsigned char* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    #define P(xx, yy) ((int)in[clampi_d((yy), 0, h - 1) * w + clampi_d((xx), 0, w - 1)])
    int gx = -P(x - 1, y - 1) + P(x + 1, y - 1)
             - 2 * P(x - 1, y) + 2 * P(x + 1, y)
             - P(x - 1, y + 1) + P(x + 1, y + 1);
    int gy = -P(x - 1, y - 1) - 2 * P(x, y - 1) - P(x + 1, y - 1)
             + P(x - 1, y + 1) + 2 * P(x, y + 1) + P(x + 1, y + 1);
    #undef P
    float mag = sqrtf((float)(gx * gx + gy * gy));
    if (mag > 255.0f) mag = 255.0f;
    out[y * w + x] = (unsigned char)(mag + 0.5f);
}

// ---------------------------------------------------------------- Generic host wrapper (Member A)
// malloc -> H2D copy -> launch -> D2H copy -> free, timing each phase with CUDA events.
template <typename Launch>
static GpuTimes run_gpu(const unsigned char* h_in, size_t in_bytes,
                        unsigned char* h_out, size_t out_bytes, Launch launch) {
    unsigned char *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, in_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));

    cudaEvent_t e0, e1, e2, e3;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2)); CUDA_CHECK(cudaEventCreate(&e3));

    CUDA_CHECK(cudaEventRecord(e0));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e1));
    launch(d_in, d_out);
    CUDA_CHECK(cudaGetLastError());                      // catches bad launch configuration
    CUDA_CHECK(cudaEventRecord(e2));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(e3));
    CUDA_CHECK(cudaEventSynchronize(e3));                // wait for everything to finish

    GpuTimes t;
    CUDA_CHECK(cudaEventElapsedTime(&t.h2d_ms,    e0, e1));
    CUDA_CHECK(cudaEventElapsedTime(&t.kernel_ms, e1, e2));
    CUDA_CHECK(cudaEventElapsedTime(&t.d2h_ms,    e2, e3));

    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3);
    cudaFree(d_in); cudaFree(d_out);
    return t;
}

static dim3 gridFor(int w, int h) { return dim3((w + BLOCK - 1) / BLOCK, (h + BLOCK - 1) / BLOCK); }

// ---------------------------------------------------------------- Public functions
GpuTimes grayscale_gpu(const unsigned char* h_rgb, unsigned char* h_gray, int w, int h) {
    dim3 block(BLOCK, BLOCK), grid = gridFor(w, h);
    return run_gpu(h_rgb, (size_t)w * h * 3, h_gray, (size_t)w * h,
        [&](const unsigned char* d_in, unsigned char* d_out) {
            grayscaleKernel<<<grid, block>>>(d_in, d_out, w, h);
        });
}

GpuTimes blur_gpu_global(const unsigned char* h_gray, unsigned char* h_out, int w, int h) {
    dim3 block(BLOCK, BLOCK), grid = gridFor(w, h);
    return run_gpu(h_gray, (size_t)w * h, h_out, (size_t)w * h,
        [&](const unsigned char* d_in, unsigned char* d_out) {
            blurGlobalKernel<<<grid, block>>>(d_in, d_out, w, h);
        });
}

GpuTimes blur_gpu_shared(const unsigned char* h_gray, unsigned char* h_out, int w, int h) {
    dim3 block(TILE, TILE), grid = gridFor(w, h);
    return run_gpu(h_gray, (size_t)w * h, h_out, (size_t)w * h,
        [&](const unsigned char* d_in, unsigned char* d_out) {
            blurSharedKernel<<<grid, block>>>(d_in, d_out, w, h);
        });
}

GpuTimes sobel_gpu(const unsigned char* h_gray, unsigned char* h_out, int w, int h) {
    dim3 block(BLOCK, BLOCK), grid = gridFor(w, h);
    return run_gpu(h_gray, (size_t)w * h, h_out, (size_t)w * h,
        [&](const unsigned char* d_in, unsigned char* d_out) {
            sobelKernel<<<grid, block>>>(d_in, d_out, w, h);
        });
}
