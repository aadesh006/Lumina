#include "kernels.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

//define constants so the compiler knows the size of our Shared Memory array
#define TILE_SIZE 16
#define RADIUS 3
#define SHARED_WIDTH (TILE_SIZE + 2 * RADIUS)

__global__ void blurKernelShared(const unsigned char* input, unsigned char* output, int width, int height) {
    
    __shared__ unsigned char s_tile[SHARED_WIDTH][SHARED_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_SIZE + tx;
    int y = blockIdx.y * TILE_SIZE + ty;

    int num_elements = SHARED_WIDTH * SHARED_WIDTH;
    int tid = ty * TILE_SIZE + tx; // My local thread ID (0 to 255)

    for (int i = tid; i < num_elements; i += (TILE_SIZE * TILE_SIZE)) {
        int s_row = i / SHARED_WIDTH;
        int s_col = i % SHARED_WIDTH;
        
        // Map shared coordinate back to global image coordinate
        int g_row = blockIdx.y * TILE_SIZE - RADIUS + s_row;
        int g_col = blockIdx.x * TILE_SIZE - RADIUS + s_col;

        // Boundary check for the Global Image
        if (g_row >= 0 && g_row < height && g_col >= 0 && g_col < width) {
            s_tile[s_row][s_col] = input[g_row * width + g_col];
        } else {
            s_tile[s_row][s_col] = 0;
        }
    }

    __syncthreads();

    if (x < width && y < height) {
        int pixVal = 0;
        int pixelsCounted = 0;

        for (int blurRow = -RADIUS; blurRow <= RADIUS; ++blurRow) {
            for (int blurCol = -RADIUS; blurCol <= RADIUS; ++blurCol) {

                int s_y = ty + RADIUS + blurRow;
                int s_x = tx + RADIUS + blurCol;
                
                pixVal += s_tile[s_y][s_x];
                pixelsCounted++;
            }
        }

        output[y * width + x] = (unsigned char)(pixVal / pixelsCounted);
    }
}

void launchBlurKernelAsync(const unsigned char* d_input, unsigned char* d_output, int width, int height, int blurSize, cudaStream_t stream) {
}

void launchBlurKernel(const unsigned char* h_input, unsigned char* h_output, int width, int height, int blurSize) {
    size_t numPixels = width * height; 
    unsigned char *d_input, *d_output;

    CHECK_CUDA(cudaMalloc((void**)&d_input, numPixels));
    CHECK_CUDA(cudaMalloc((void**)&d_output, numPixels));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, numPixels, cudaMemcpyHostToDevice));

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    printf("Launching SHARED MEMORY Blur Kernel (Radius %d)...\n", RADIUS);

    // Call the new shared memory kernel
    blurKernelShared<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_output, d_output, numPixels, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}