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

//THE BLUR KERNEL (Runs on GPU)
__global__ void blurKernel(const unsigned char* input, unsigned char* output, int width, int height, int blurSize) {
    //Get Global Thread ID
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //Boundary Guard: Are we inside the image?
    if (x < width && y < height) {
        int pixVal = 0;
        int pixelsCounted = 0;

        //Stencil Loop: Look at neighbors
        for (int blurRow = -blurSize; blurRow <= blurSize; ++blurRow) {
            for (int blurCol = -blurSize; blurCol <= blurSize; ++blurCol) {
                
                int curRow = y + blurRow;
                int curCol = x + blurCol;

                //Inner Boundary Guard: Is the neighbor inside the image?
                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                    pixVal += input[curRow * width + curCol];
                    pixelsCounted++;
                }
            }
        }

        //Write the average value to the output
        output[y * width + x] = (unsigned char)(pixVal / pixelsCounted);
    }
}

//THE LAUNCHER (Runs on CPU)
void launchBlurKernel(const unsigned char* h_input, unsigned char* h_output, int width, int height, int blurSize) {
    size_t numPixels = width * height; 
    unsigned char *d_input, *d_output;

    CHECK_CUDA(cudaMalloc((void**)&d_input, numPixels));
    CHECK_CUDA(cudaMalloc((void**)&d_output, numPixels));

    // Copy Data to GPU
    CHECK_CUDA(cudaMemcpy(d_input, h_input, numPixels, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    printf("Launching Blur Kernel (Radius %d) on %dx%d grid...\n", blurSize, gridSize.x, gridSize.y);

    // Launch Kernel
    blurKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, blurSize);
    
    // Catch execution errors
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy Data back to CPU
    CHECK_CUDA(cudaMemcpy(h_output, d_output, numPixels, cudaMemcpyDeviceToHost));

    // Free VRAM
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}