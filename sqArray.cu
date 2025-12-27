#include <stdio.h>
#include <cuda_runtime.h>

// 1. The Kernel: This code runs on the GPU
__global__ void squareArray(float *d_out, float *d_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = d_in[idx] * d_in[idx];
    }
}

int main() {
    const int N = 5;
    const int size = N * sizeof(float);

    // Host (CPU) data
    float h_in[N] = {1, 2, 3, 4, 5};
    float h_out[N];

    // 2. Device (GPU) pointers
    float *d_in, *d_out;

    // 3. Allocate GPU Memory
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    // 4. Copy data from Host to Device
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // 5. Launch Kernel: 1 block with N threads
    squareArray<<<1, N>>>(d_out, d_in, N);

    // 6. Copy result back from Device to Host
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < N; i++) {
        printf("%.1f squared is %.1f\n", h_in[i], h_out[i]);
    }

    // 7. Cleanup
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}