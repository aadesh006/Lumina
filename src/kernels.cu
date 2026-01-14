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

__global__ void grayscaleKernel(const unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int grayIdx = y * width + x;
        int rgbIdx = grayIdx * 3;

        unsigned char r = input[rgbIdx];
        unsigned char g = input[rgbIdx + 1];
        unsigned char b = input[rgbIdx + 2];

        output[grayIdx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// THE BLUR KERNEL
__global__ void blurKernel(const unsigned char* input, unsigned char* output, int width, int height, int blurSize) {
    
    //Global Thread ID
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        
        int pixVal = 0;
        int pixelsCounted = 0;

        //The Stencil Loop
        for (int blurRow = -blurSize; blurRow <= blurSize; ++blurRow) {
            for (int blurCol = -blurSize; blurCol <= blurSize; ++blurCol) {
                
                int curRow = y + blurRow;
                int curCol = x + blurCol;

                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                    // Read global memory
                    pixVal += input[curRow * width + curCol];
                    pixelsCounted++;
                }
            }
        }

        //Write Average (Sum / Count)
        output[y * width + x] = (unsigned char)(pixVal / pixelsCounted);
    }
}

void runCudaProcess(const unsigned char* h_input, unsigned char* h_output, int width, int height) {
    size_t numPixels = width * height;
    unsigned char *d_input, *d_output;

    // 1. Allocate with Error Checking
    CHECK_CUDA(cudaMalloc((void**)&d_input, numPixels * 3));
    CHECK_CUDA(cudaMalloc((void**)&d_output, numPixels));

    // 2. Copy with Error Checking
    CHECK_CUDA(cudaMemcpy(d_input, h_input, numPixels * 3, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    printf("Launching Grid: %dx%d blocks\n", gridSize.x, gridSize.y);

    // 3. Launch Kernel
    grayscaleKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    // 4. Check for Kernel Launch Errors
    CHECK_CUDA(cudaGetLastError());
    
    // 5. Force GPU to finish (Synchronize) to catch execution errors
    CHECK_CUDA(cudaDeviceSynchronize());

    // 6. Copy Back
    CHECK_CUDA(cudaMemcpy(h_output, d_output, numPixels, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}